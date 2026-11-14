import numpy as np
import pytest

from HAOA.benchmark_functions.ackley import ackley
from HAOA.benchmark_functions.griewank import griewank
from HAOA.benchmark_functions.rastrigin import rastrigin
from HAOA.benchmark_functions.rosenbrock import rosenbrock
from HAOA.benchmark_functions.sphere import sphere


@pytest.mark.parametrize(
    "function",
    [sphere, rastrigin, ackley, griewank, rosenbrock],
)
def test_known_global_optimum_is_zero(function):
    optimum = np.ones(5) if function is rosenbrock else np.zeros(5)
    assert float(function(optimum)) == pytest.approx(0.0, abs=1e-12)


def test_rosenbrock_minimum_at_all_ones():
    assert rosenbrock(np.ones(5)) == pytest.approx(0.0)


def test_benchmark_functions_accept_numpy_vectors():
    vector = np.array([0.5, -0.25, 0.75])
    for function in (sphere, rastrigin, ackley, griewank, rosenbrock):
        value = function(vector)
        assert np.isscalar(value)
        assert np.isfinite(value)
