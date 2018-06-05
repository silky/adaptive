# -*- coding: utf-8 -*-
from collections import OrderedDict, defaultdict
from copy import copy, deepcopy
import itertools
from math import sqrt
import sys

import numpy as np
from scipy import interpolate
from sortedcontainers import SortedDict

from ..notebook_integration import ensure_holoviews
from .base_learner import BaseLearner
from .learner2D import default_loss, choose_point_in_triangle, areas


def standard_error(lst):
    n = len(lst)
    if n < 2:
        return sys.float_info.max
    sum_f_sq = sum(x**2 for x in lst)
    mean = sum(x for x in lst) / n
    std = sqrt((sum_f_sq - n * mean**2) / (n - 1))
    return std / sqrt(n)


class AverageLearner2D(BaseLearner):
    def __init__(self, function, bounds, loss_per_triangle=None):
        self.ndim = len(bounds)
        self._vdim = None
        self.loss_per_triangle = loss_per_triangle or default_loss
        self.bounds = tuple((float(a), float(b)) for a, b in bounds)
        self._data = defaultdict(list)
        self._stack = OrderedDict()
        self._interp = set()

        self.xy_mean = np.mean(self.bounds, axis=1)
        self._xy_scale = np.ptp(self.bounds, axis=1)
        self.aspect_ratio = 1

        self._bounds_points = list(itertools.product(*bounds))
        self._stack.update({p: np.inf for p in self._bounds_points})
        self.function = function
        self._ip = self._ip_combined = None
        self._loss = np.inf

        self.stack_size = 10

    @property
    def data(self):
        return {k: sum(v) / len(v) for k, v in self._data.items()}

    @property
    def data_sem(self):
        return {k: standard_error(v) for k, v in self._data.items()}

    @property
    def xy_scale(self):
        xy_scale = self._xy_scale
        if self.aspect_ratio == 1:
            return xy_scale
        else:
            return np.array([xy_scale[0], xy_scale[1] / self.aspect_ratio])

    def _scale(self, points):
        points = np.asarray(points, dtype=float)
        return (points - self.xy_mean) / self.xy_scale

    def _unscale(self, points):
        points = np.asarray(points, dtype=float)
        return points * self.xy_scale + self.xy_mean

    @property
    def npoints(self):
        return len(self.data)

    @property
    def vdim(self):
        if self._vdim is None and self.data:
            try:
                value = next(iter(self.data.values()))
                self._vdim = len(value)
            except TypeError:
                self._vdim = 1
        return self._vdim if self._vdim is not None else 1

    @property
    def bounds_are_done(self):
        return not any((p in self._interp or p in self._stack)
                       for p in self._bounds_points)

    def data_combined(self):
        # Interpolate the unfinished points
        data_combined = copy(self.data)
        if self._interp:
            points_interp = list(self._interp)
            if self.bounds_are_done:
                values_interp = self.ip()(self._scale(points_interp))
            else:
                # Without the bounds the interpolation cannot be done properly,
                # so we just set everything to zero.
                values_interp = np.zeros((len(points_interp), self.vdim))

            for point, value in zip(points_interp, values_interp):
                data_combined[point] = value

        return data_combined

    def ip(self):
        if self._ip is None:
            points = self._scale(list(self.data.keys()))
            values = np.array(list(self.data.values()), dtype=float)
            self._ip = interpolate.LinearNDInterpolator(points, values)
        return self._ip

    def ip_combined(self):
        if self._ip_combined is None:
            data_combined = self.data_combined()
            points = self._scale(list(data_combined.keys()))
            values = np.array(list(data_combined.values()), dtype=float)
            self._ip_combined = interpolate.LinearNDInterpolator(points,
                                                                 values)
        return self._ip_combined

    def _tell(self, point, value):
        point = tuple(point)

        if value is None:
            self._interp.add(point)
            self._ip_combined = None
        else:
            self._data[point].append(value)
            self._interp.discard(point)
            self._ip = None

        self._stack.pop(point, None)

    def _fill_stack(self, stack_till=1):
        if len(self.data) + len(self._interp) < self.ndim + 1:
            raise ValueError("too few points...")

        # Interpolate
        ip = self.ip_combined()

        losses = self.loss_per_triangle(ip)

        points_new = []
        losses_new = []
        for j, _ in enumerate(losses):
            jsimplex = np.argmax(losses)
            triangle = ip.tri.points[ip.tri.vertices[jsimplex]]
            point_new = choose_point_in_triangle(triangle, max_badness=5)
            point_new = tuple(self._unscale(point_new))
            loss_new = abs(losses[jsimplex])

            points_new.append(point_new)
            losses_new.append(loss_new)

            self._stack[point_new] = loss_new

            if len(self._stack) >= stack_till:
                break
            else:
                losses[jsimplex] = -np.inf

        return points_new, losses_new

    def ask(self, n, add_data=True):
        # Even if add_data is False we add the point such that _fill_stack
        # will return new points, later we remove these points if needed.
        if len(self._stack) < 1:
            self._fill_stack(self.stack_size)

        stack = {**self._stack, **self.data_sem}
        points, loss_improvements = zip(*sorted(stack.items(),
                                                key=lambda x: -x[1]))

        n_left = n - len(points)
        self.tell(points[:n], itertools.repeat(None))

        while n_left > 0:
            # The while loop is needed because `stack_till` could be larger
            # than the number of triangles between the points. Therefore
            # it could fill up till a length smaller than `stack_till`.
            new_points, new_loss_improvements = self._fill_stack(
                stack_till=max(n_left, self.stack_size))
            self.tell(new_points[:n_left], itertools.repeat(None))
            n_left -= len(new_points)

            points += new_points
            loss_improvements += new_loss_improvements

        if not add_data:
            self._stack = OrderedDict(zip(points[:self.stack_size],
                                          loss_improvements))
            for point in points[:n]:
                self._interp.discard(point)

        return points[:n], loss_improvements[:n]

    def loss(self, real=True):
        if not self.bounds_are_done:
            return np.inf
        ip = self.ip() if real else self.ip_combined()
        losses = self.loss_per_triangle(ip)
        self._loss = losses.max()
        return self._loss

    def remove_unfinished(self):
        self._interp = set()
        for p in self._bounds_points:
            if p not in self.data:
                self._stack[p] = np.inf

    def plot(self, n=None, tri_alpha=0):
        hv = ensure_holoviews()
        if self.vdim > 1:
            raise NotImplemented('holoviews currently does not support',
                                 '3D surface plots in bokeh.')
        x, y = self.bounds
        lbrt = x[0], y[0], x[1], y[1]

        if len(self.data) >= 4:
            ip = self.ip()

            if n is None:
                # Calculate how many grid points are needed.
                # factor from A=√3/4 * a² (equilateral triangle)
                n = int(0.658 / sqrt(areas(ip).min()))
                n = max(n, 10)

            x = y = np.linspace(-0.5, 0.5, n)
            z = ip(x[:, None], y[None, :] * self.aspect_ratio).squeeze()

            im = hv.Image(np.rot90(z), bounds=lbrt)

            if tri_alpha:
                points = self._unscale(ip.tri.points[ip.tri.vertices])
                points = np.pad(points[:, [0, 1, 2, 0], :],
                                pad_width=((0, 0), (0, 1), (0, 0)),
                                mode='constant',
                                constant_values=np.nan).reshape(-1, 2)
                tris = hv.EdgePaths([points])
            else:
                tris = hv.EdgePaths([])
        else:
            im = hv.Image([], bounds=lbrt)
            tris = hv.EdgePaths([])

        im_opts = dict(cmap='viridis')
        tri_opts = dict(line_width=0.5, alpha=tri_alpha)
        no_hover = dict(plot=dict(inspection_policy=None, tools=[]))

        return im.opts(style=im_opts) * tris.opts(style=tri_opts, **no_hover)
