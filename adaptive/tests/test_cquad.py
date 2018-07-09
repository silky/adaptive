# -*- coding: utf-8 -*-
from functools import partial
from operator import attrgetter

import numpy as np
import pytest
from ..learner import IntegratorLearner
from ..learner.integrator_learner import DivergentIntegralError
from ..learner.integrator_coeffs import ns
from .algorithm_4 import algorithm_4, f0, f7, f21, f24, f63, fdiv
from .algorithm_4 import DivergentIntegralError as A4DivergentIntegralError

eps = np.spacing(1)


def rolling_shuffle(nums, size):
    for i in range(len(nums) - size):
        x = nums[i:i+size+1]
        random.shuffle(x)
        nums[i:i+size+1] = x
    return nums


def run_integrator_learner(f, a, b, tol, n):
    learner = IntegratorLearner(f, bounds=(a, b), tol=tol)
    for _ in range(n):
        points, _ = learner.ask(1)
        learner.tell(points, map(learner.function, points))
    return learner


def equal_ival(ival, other, *, verbose=False):
    """Note: Implementing __eq__ breaks SortedContainers in some way."""
    if ival.depth_complete is None:
        if verbose:
            print('Interval {} is not complete.'.format(ival))
        return False

    slots = set(ival.__slots__).intersection(other.__slots__)
    same_slots = []
    for s in slots:
        a = getattr(ival, s)
        b = getattr(other, s)
        is_equal = np.allclose(a, b, rtol=0, atol=eps, equal_nan=True)
        if verbose and not is_equal:
            print('ival.{} - other.{} = {}'.format(s, s, a - b))
        same_slots.append(is_equal)

    return all(same_slots)


def equal_ivals(ivals, other, *, verbose=False):
    """Note: `other` is a list of ivals."""
    if len(ivals) != len(other):
        if verbose:
            print('len(ivals)={} != len(other)={}'.format(
                len(ivals), len(other)))
        return False

    ivals = [sorted(i, key=attrgetter('a')) for i in [ivals, other]]
    return all(equal_ival(ival, other_ival, verbose=verbose)
               for ival, other_ival in zip(*ivals))


def same_ivals(f, a, b, tol):
        igral, err, n, ivals = algorithm_4(f, a, b, tol)

        learner = run_integrator_learner(f, a, b, tol, n)

        # This will only show up if the test fails, anyway
        print('igral difference', learner.igral-igral,
              'err difference', learner.err - err)

        return equal_ivals(learner.ivals, ivals, verbose=True)

# XXX: This *should* pass (https://gitlab.kwant-project.org/qt/adaptive/issues/84)
@pytest.mark.xfail
def test_that_gives_same_intervals_as_reference_implementation():
    for i, args in enumerate([[f0, 0, 3, 1e-5],
                              [f7, 0, 1, 1e-6],
                              [f21, 0, 1, 1e-3],
                              [f24, 0, 3, 1e-3]]):
        assert same_ivals(*args), 'Function {}'.format(i)


@pytest.mark.xfail
def test_machine_precision():
    f, a, b, tol = [partial(f63, alpha=0.987654321, beta=0.45), 0, 1, 1e-10]
    igral, err, n, ivals = algorithm_4(f, a, b, tol)

    learner = run_integrator_learner(f, a, b, tol, n)

    print('igral difference', learner.igral-igral,
          'err difference', learner.err - err)

    assert equal_ivals(learner.ivals, ivals, verbose=True)


def test_machine_precision2():
    f, a, b, tol = [partial(f63, alpha=0.987654321, beta=0.45), 0, 1, 1e-10]
    igral, err, n, ivals = algorithm_4(f, a, b, tol)

    learner = run_integrator_learner(f, a, b, tol, n)

    np.testing.assert_almost_equal(igral, learner.igral)
    np.testing.assert_almost_equal(err, learner.err)


def test_divergence():
    """This function should raise a DivergentIntegralError."""
    f, a, b, tol = fdiv, 0, 1, 1e-6
    with pytest.raises(A4DivergentIntegralError) as e:
        igral, err, n, ivals = algorithm_4(f, a, b, tol)

    n = e.value.nr_points

    with pytest.raises(DivergentIntegralError):
        run_integrator_learner(f, a, b, tol, n)


def test_choosing_and_adding_points_one_by_one():
    learner = IntegratorLearner(f24, bounds=(0, 3), tol=1e-10)
    for _ in range(1000):
        xs, _ = learner.ask(1)
        for x in xs:
            learner.tell(x, learner.function(x))


def test_choosing_and_adding_multiple_points_at_once():
    learner = IntegratorLearner(f24, bounds=(0, 3), tol=1e-10)
    xs, _ = learner.ask(100)
    for x in xs:
        learner.tell(x, learner.function(x))


def test_adding_points_and_skip_one_point():
    learner = IntegratorLearner(f24, bounds=(0, 3), tol=1e-10)
    xs, _ = learner.ask(17)
    skip_x = xs[1]

    for x in xs:
        if x != skip_x:
            learner.tell(x, learner.function(x))

    for i in range(1000):
        xs, _ = learner.ask(1)
        for x in xs:
            if x != skip_x:
                learner.tell(x, learner.function(x))

    # Now add the point that was skipped
    learner.tell(skip_x, learner.function(skip_x))

    # Create a learner with the same number of points, which should
    # give an identical igral value.
    learner2 = IntegratorLearner(f24, bounds=(0, 3), tol=1e-10)
    for i in range(1017):
        xs, _ = learner2.ask(1)
        for x in xs:
            learner2.tell(x, learner2.function(x))

    np.testing.assert_almost_equal(learner.igral, learner2.igral)



# XXX: This *should* pass (https://gitlab.kwant-project.org/qt/adaptive/issues/84)
@pytest.mark.xfail
def test_tell_in_random_order(first_add_33=False):
    from operator import attrgetter
    import random
    tol = 1e-10
    for f, a, b in ([f0, 0, 3],
                    [f21, 0, 1],
                    [f24, 0, 3],
                    [f7, 0, 1],
                    ):
        learners = []

        for shuffle in [True, False]:
            l = IntegratorLearner(f, bounds=(a, b), tol=tol)

            if first_add_33:
                xs, _ = l.ask(33)
                for x in xs:
                    l.tell(x, f(x))

            xs, _ = l.ask(10000)

            if shuffle:
                random.shuffle(xs)
            for x in xs:
                l.tell(x, f(x))

            learners.append(l)

        # Check whether the points of the learners are identical
        assert set(learners[0].done_points) == set(learners[1].done_points)

        # Test whether approximating_intervals gives a complete set of intervals
        for l in learners:
            ivals = sorted(l.approximating_intervals, key=lambda l: l.a)
            for i in range(len(ivals) - 1):
                assert ivals[i].b == ivals[i + 1].a, (ivals[i], ivals[i + 1])

        # Test if approximating_intervals is the same for random order of adding the point
        ivals = [sorted(ival, key=attrgetter('a')) for ival in
                 [l.approximating_intervals for l in learners]]
        assert all(ival.a == other_ival.a for ival, other_ival in zip(*ivals))

        # Test if the approximating_intervals are the same
        ivals = [set((i.a, i.b) for i in l.approximating_intervals)
                 for l in learners]
        assert ivals[0] == ivals[1]

        # Test whether the igral is identical
        assert np.allclose(learners[0].igral, learners[1].igral), f

        # Compare if the errors are in line with the sequential case
        igral, err, *_ = algorithm_4(f, a, b, tol=tol)
        assert all((l.err + err >= abs(l.igral - igral)) for l in learners)

        # Check that the errors are finite
        for l in learners:
            assert np.isfinite(l.err)


def test_tell_in_random_order_first_add_33():
    test_tell_in_random_order(first_add_33=True)


def test_approximating_intervals():
    import random
    learner = IntegratorLearner(f24, bounds=(0, 3), tol=1e-10)

    xs, _ = learner.ask(10000)
    random.shuffle(xs)
    for x in xs:
        learner.tell(x, f24(x))

    ivals = sorted(learner.approximating_intervals, key=lambda l: l.a)
    for i in range(len(ivals) - 1):
        assert ivals[i].b == ivals[i + 1].a, (ivals[i], ivals[i + 1])


# XXX: This *should* pass (https://gitlab.kwant-project.org/qt/adaptive/issues/84)
@pytest.mark.xfail
def test_removed_choose_mutiple_points_at_once():
    """Given that a high-precision interval that was split into 2 low-precision ones,
       we should use the high-precision interval.
    """
    learner = IntegratorLearner(np.exp, bounds=(0, 1), tol=1e-15)
    n = ns[-1] + 2 * (ns[0] - 2)  # first + two children (33+6=39)
    xs, _ = learner.ask(n)
    for x in xs:
        learner.tell(x, learner.function(x))
    assert list(learner.approximating_intervals)[0] == learner.first_ival


def test_removed_ask_one_by_one():
    with pytest.raises(RuntimeError):
        # This test should raise because integrating np.exp should be done
        # after the 33th point
        learner = IntegratorLearner(np.exp, bounds=(0, 1), tol=1e-15)
        n = ns[-1] + 2 * (ns[0] - 2)  # first + two children (33+6=39)
        for _ in range(n):
            xs, _ = learner.ask(1)
            for x in xs:
                learner.tell(x, learner.function(x))
