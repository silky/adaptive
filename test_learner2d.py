# -*- coding: utf-8 -*-
# test for adaptive package
# preamble

from functools import partial
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import special, interpolate
from time import sleep

import sys
import adaptive

executor = ProcessPoolExecutor(max_workers=1)

def peak_finder(ip):
    """Search for peaks following 3 strategies.

    First, search for vertices that are highest amongst more than one triangle.
    Second, filter those peaks with values lower than a threshold.
    Third, filter those peaks with gradients higher than a threshold.
    """
    tri = ip.tri
    vs = ip.values.ravel()
    vertices = tri.vertices
    n_tri, n_vertices = vertices.shape

    gradients = interpolate.interpnd.estimate_gradients_2d_global(
        tri, vs, tol=1e-6)
    p = tri.points[vertices]
    v = vs[vertices]

    # search for vertices that are highest
    greatest_vs = vertices[np.arange(n_tri), np.argmax(v, axis=-1)]

    # filter vertices that are the highest in at least N triangles
    min_repetitions = 4
    unique_elements, counts = np.unique(greatest_vs, return_counts=True)
    repeated = unique_elements[counts>min_repetitions]

    # filter vertices with low values, below a threshold
    repeated = repeated[vs[repeated]>1]

    return tri.points[repeated]

def cauchy(xy, xy0, radius):
    xy0 = np.atleast_2d(xy0)
    gamma = radius
    num_peaks = xy0.shape[0]
    x, y = xy
    result = 0
    for i in range(num_peaks):
        x0, y0 = xy0[i]
        result += (gamma / (2 * np.pi) /
                ((x-x0)**2 + (y-y0)**2 + gamma**2)**1.5)
    return result

def normal(xy, xy0, radius):
    xy0 = np.atleast_2d(xy0)
    sigma = radius
    num_peaks = xy0.shape[0]
    x, y = xy
    result = 0
    for i in range(num_peaks):
        x0, y0 = xy0[i]
        result += (1 / (2 * np.pi * sigma**2) *
                np.exp(-((x-x0) ** 2 + (y-y0) ** 2) /
                        (2 * sigma**2)))
    return result

def test_peak_finder():
    """Find peaks in an area.

    The test resides on passing a function with randomly positioned peaks,
    of a small width. This width is defined as a fraction of the total area.
    This fraction is the number of initial triangles (2*min_resolution**2),
    times a discovery_factor. The discovery_factor controls how hard to find
    are the peaks.
    """
    bounds = [(-1, 1), (-1, 1)]
    total_area = (bounds[0][1] - bounds[0][0]) * (bounds[1][1] - bounds[1][0])
    np.random.seed(0)

    # minimum linear resolution
    min_resolution = 5
    discovery_factor = 5
    min_area = (total_area / min_resolution ** 2) / discovery_factor
    radius = (min_area / np.pi) ** 0.5
    tol = min_area / 2
    # random peaks
    box = (1-4*radius) # ensure peaks inside the area
    num_peaks = 1
    xy0 = np.ndarray((num_peaks, 2))
    for i in range(num_peaks):
        xy0[i] = (2*np.random.rand(2) - 1) * box

    for func in [partial(cauchy, xy0=xy0, radius=radius),
                 partial(normal, xy0=xy0, radius=radius)]:
        learner = adaptive.learner.Learner2D(func, bounds=bounds)
        runner = adaptive.Runner(learner, executor=executor,
                                goal=lambda l: l.loss() < tol)
        sleep(0.5)
        while runner.task.done() is False:
            sleep(0.5)
            print(runner.learner.n)

        found_peaks = learner.unscale(peak_finder(learner.ip()))
        # peaks found can be nearly degenerate, and the number can be
        # greater than num_peaks
        # The number of peaks found can be smaller than num_peaks, if some
        # peak in xy0 is nearly degenerate, or if some peak is not found,
        # in which case the test fails
        deviations = np.ndarray(num_peaks)
        for i in range(num_peaks):
            deviations[i] = np.linalg.norm(xy0[i] - found_peaks, axis=-1).min()

        assert np.all(deviations < radius)