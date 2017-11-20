# -*- coding: utf-8 -*-
import collections
from copy import deepcopy
import itertools

import holoviews as hv
import numpy as np
from scipy import interpolate, spatial

from .base_learner import BaseLearner
from .utils import restore


# Learner2D and helper functions.

def deviations(ip):
    values = ip.values / (ip.values.ptp() or 1)
    gradients = interpolate.interpnd.estimate_gradients_2d_global(
        ip.tri, values, tol=1e-6)

    p = ip.tri.points[ip.tri.vertices]
    vs = values[ip.tri.vertices]
    gs = gradients[ip.tri.vertices]

    def deviation(p, v, g):
        dev = 0
        for j in range(3):
            vest = v[:, j, None] + ((p[:, :, :] - p[:, j, None, :]) *
                                    g[:, j, None, :]).sum(axis=-1)
            dev += abs(vest - v).max(axis=1)
        return dev

    n_levels = vs.shape[2]
    devs = [deviation(p, vs[:, :, i], gs[:, :, i]) for i in range(n_levels)]
    return devs


def areas(ip):
    p = ip.tri.points[ip.tri.vertices]
    q = p[:, :-1, :] - p[:, -1, None, :]
    areas = abs(q[:, 0, 0] * q[:, 1, 1] - q[:, 0, 1] * q[:, 1, 0]) / 2
    return areas


def _default_loss_per_triangle(ip):
    devs = deviations(ip)
    area_per_triangle = np.sqrt(areas(ip))
    losses = np.sum([dev * area_per_triangle for dev in devs], axis=0)
    return losses


class Learner2D(BaseLearner):
    """Learns and predicts a function 'f: ℝ^2 → ℝ^N'.

    Parameters
    ----------
    function : callable
        The function to learn. Must take a tuple of two real
        parameters and return a real number.
    bounds : list of 2-tuples
        A list ``[(a1, b1), (a2, b2)]`` containing bounds,
        one per dimension.
    loss_per_triangle : callable, optional
        A function that returns the loss for every triangle.
        If not provided, then a default is used, which uses
        the deviation from a linear estimate, as well as
        triangle area, to determine the loss. See the notes
        for more details.


    Attributes
    ----------
    points_combined
        Sample points so far including the unknown interpolated ones.
    values_combined
        Sampled values so far including the unknown interpolated ones.
    points
        Sample points so far with real results.
    values
        Sampled values so far with real results.

    Notes
    -----
    Adapted from an initial implementation by Pauli Virtanen.

    The sample points are chosen by estimating the point where the
    linear and cubic interpolants based on the existing points have
    maximal disagreement. This point is then taken as the next point
    to be sampled.

    In practice, this sampling protocol results to sparser sampling of
    smooth regions, and denser sampling of regions where the function
    changes rapidly, which is useful if the function is expensive to
    compute.

    This sampling procedure is not extremely fast, so to benefit from
    it, your function needs to be slow enough to compute.

    'loss_per_triangle' takes a single parameter, 'ip', which is a
    `scipy.interpolate.LinearNDInterpolator`. You can use the
    *undocumented* attributes 'tri' and 'values' of 'ip' to get a
    `scipy.spatial.Delaunay` and a vector of function values.
    These can be used to compute the loss. The functions
    `adaptive.learner.learner2D.areas` and
    `adaptive.learner.learner2D.deviations` to calculate the
    areas and deviations from a linear interpolation
    over each triangle.
    """

    def __init__(self, function, bounds, loss_per_triangle=None):
        self.ndim = len(bounds)
        self._vdim = None
        self.loss_per_triangle = loss_per_triangle or _default_loss_per_triangle
        self.bounds = tuple((float(a), float(b)) for a, b in bounds)
        self.data = collections.OrderedDict()
        self.data_combined = collections.OrderedDict()
        self._stack = collections.OrderedDict()
        self._interp = set()

        xy_mean = np.mean(self.bounds, axis=1)
        xy_scale = np.ptp(self.bounds, axis=1)

        def scale(points):
            points = np.asarray(points)
            return (points - xy_mean) / xy_scale

        def unscale(points):
            points = np.asarray(points)
            return points * xy_scale + xy_mean

        self.scale = scale
        self.unscale = unscale

        self._bounds_points = list(itertools.product(*bounds))

        self._tri = None
        self.tri_combined = spatial.Delaunay(self.scale(self._bounds_points),
                                             incremental=True,
                                             qhull_options='Q11 QJ')

        for point in self._bounds_points:
            self.data_combined[point] = None
            self._stack[point] = np.inf
            self._interp.add(point)

        self.function = function

        self._ip = self._ip_combined = None

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
    def points_combined(self):
        return np.array(list(self.data_combined.keys()))

    @property
    def values_combined(self):
        return np.array(list(self.data_combined.values()))

    @property
    def points(self):
        return np.array(list(self.data.keys()))

    @property
    def values(self):
        return np.array(list(self.data.values()))

    @property
    def bounds_are_done(self):
        return not any(p in self._interp for p in self._bounds_points)

    @property
    def tri(self):
        if self._tri is None:
            self._tri = spatial.Delaunay(self.scale(self.points),
                                         incremental=True,
                                         qhull_options='Q11 QJ')
        return self._tri

    @property
    def bounds_are_done(self):
        return not any((p in self._interp or p in self._stack)
                       for p in self._bounds_points)

    def ip(self):
        if self._ip is None:
            points = self.scale(self.points)
            self._ip = interpolate.LinearNDInterpolator(points, self.values)
        return self._ip

    def ip_combined(self):
        if self._ip_combined is None:
            points = self.scale(self.points_combined)
            values = self.values_combined

            # Interpolate the unfinished points
            if self._interp:
                points_interp = list(self._interp)
                if self.bounds_are_done:
                    values_interp = self.ip()(self.scale(points_interp))
                else:
                    values_interp = np.zeros((len(points_interp), self.vdim))

            for point, value in zip(points_interp, values_interp):
                assert point in self.data_combined
                self.data_combined[point] = value

            self._ip_combined = interpolate.LinearNDInterpolator(self.tri_combined,
                                                                 self.values_combined)
        return self._ip_combined

    def add_point(self, point, value):
        point = tuple(point)

        new_point = point not in self.data_combined
        self.data_combined[point] = value
        if value is None or new_point:
            if point not in self._bounds_points:
                self.tri_combined.add_points([self.scale(point)])

        if value is None:
            self._interp.add(point)
        else:
            if self.bounds_are_done:
                assert point not in self.data  # XXX: this has to be
                self.tri.add_points([self.scale(point)])
            self.data[point] = value
            self._interp.discard(point)

        self._stack.pop(point, None)

        # Reset the in LinearNDInterpolator objects
        self._ip = self._ip_combined = None

        # Just for debugging:
        if self._tri is not None:
            for points, tri in [(self.points, self.tri),
                                (self.points_combined, self.tri_combined)]:
                assert np.max(points - self.unscale(tri.points)) == 0
            assert len(self.points_combined) == len(self.points) + len(self._interp)

    def _fill_stack(self, stack_till=1):
        if len(self.data_combined) < len(self.bounds) + 1:
            raise ValueError("too few points...")

        # Interpolate
        ip = self.ip_combined()

        losses = self.loss_per_triangle(ip)

        for j, _ in enumerate(losses):
            jsimplex = np.argmax(losses)
            point_new = ip.tri.points[ip.tri.vertices[jsimplex]]
            point_new = self.unscale(point_new.mean(axis=-2))
            point_new = tuple(np.clip(point_new, *zip(*self.bounds)))

            # Check if it is really new
            if point_new in self.data_combined:
                # XXX: maybe check whether the point_new is not very close the another point
                losses[jsimplex] = -np.inf
                continue

            self._stack[point_new] = losses[jsimplex]

            if len(self._stack) >= stack_till:
                break
            else:
                losses[jsimplex] = -np.inf

    def _split_stack(self, n=None):
        points, loss_improvements = zip(*reversed(self._stack.items()))
        return points[:n], loss_improvements[:n]

    def _choose_and_add_points(self, n):
        points = []
        loss_improvements = []
        n_left = n
        while n_left > 0:
            # The while loop is needed because `stack_till` could be larger
            # than the number of triangles between the points. Therefore
            # it could fill up till a length smaller than `stack_till`.
            if not any(p in self._stack for p in self._bounds_points):
                self._fill_stack(stack_till=n_left)
            new_points, new_loss_improvements = self._split_stack(max(n_left, 10))
            points += new_points
            loss_improvements += new_loss_improvements
            self.add_data(new_points, itertools.repeat(None))
            n_left -= len(new_points)

        return points, loss_improvements

    def choose_points(self, n, add_data=True):
        if not add_data:
            with restore(self):
                return self._choose_and_add_points(n)
        else:
            return self._choose_and_add_points(n)

    def loss(self, real=True):
        if not self.bounds_are_done:
            return np.inf
        ip = self.ip() if real else self.ip_combined()
        losses = self.loss_per_triangle(ip)
        return losses.max()

    def remove_unfinished(self):
        self.data_combined = deepcopy(self.data)
        self.tri_combined = spatial.Delaunay(self.scale(self.points),
                                             incremental=True,
                                             qhull_options='Q11 QJ')
        self._interp = set()

    def plot(self, n_x=201, n_y=201, triangles_alpha=0):
        if self.vdim > 1:
            raise NotImplemented('holoviews currently does not support',
                                 '3D surface plots in bokeh.')
        x, y = self.bounds
        lbrt = x[0], y[0], x[1], y[1]
        if len(self.data) >= 4:
            x = np.linspace(-0.5, 0.5, n_x)
            y = np.linspace(-0.5, 0.5, n_y)
            ip = self.ip()
            z = ip(x[:, None], y[None, :])
            plot = hv.Image(np.rot90(z), bounds=lbrt)

            if triangles_alpha:
                tri_points = self.unscale(ip.tri.points[ip.tri.vertices])
                contours = hv.Contours([p for p in tri_points])
                contours = contours.opts(style=dict(alpha=triangles_alpha))

        else:
            plot = hv.Image([], bounds=lbrt)
            contours = hv.Contours([])

        return plot * contours if triangles_alpha else plot
