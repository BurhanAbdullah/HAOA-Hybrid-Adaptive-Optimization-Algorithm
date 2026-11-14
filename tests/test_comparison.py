import numpy as np
import pytest

from HAOA.algorithm.haoa import HAOA
from HAOA.comparison_algorithms.pso import PSO
from HAOA.experiments.dataset_comparison import SimplePSO


def sphere(x):
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


@pytest.mark.parametrize("optimizer_class", [PSO, SimplePSO])
def test_pso_comparison_implementations_return_valid_results(optimizer_class):
    optimizer = optimizer_class(
        objective_function=sphere,
        dimension=5,
        lower_bound=-4.0,
        upper_bound=4.0,
        population_size=10,
        max_iterations=12,
    )
    result = optimizer.optimize()

    assert np.isfinite(result["best_score"])
    assert result["best_score"] >= 0.0
    assert len(result["convergence_curve"]) == 12
    assert np.all(np.isfinite(result["convergence_curve"]))
    assert np.all(np.diff(result["convergence_curve"]) <= 1e-12)

    solution = result.get("best_solution", result.get("global_best_position"))
    if solution is not None:
        assert np.asarray(solution).shape == (5,)
        assert np.all(np.asarray(solution) >= -4.0)
        assert np.all(np.asarray(solution) <= 4.0)


@pytest.mark.parametrize("optimizer_class", [HAOA, PSO, SimplePSO])
def test_optimizers_handle_scalar_and_vector_bounds(optimizer_class):
    options = dict(
        objective_function=sphere,
        dimension=3,
        lower_bound=np.array([-1.0, -2.0, -3.0]),
        upper_bound=np.array([1.0, 2.0, 3.0]),
        population_size=8,
        max_iterations=8,
    )
    optimizer = optimizer_class(**options)
    result = optimizer.optimize()

    assert np.isfinite(result["best_score"])
    assert len(result["convergence_curve"]) == 8


def test_haoa_and_pso_are_both_compared_on_same_objective():
    haoa = HAOA(sphere, 4, -3.0, 3.0, 10, 10, random_state=7)
    pso = PSO(sphere, 4, -3.0, 3.0, 10, 10)

    haoa_result = haoa.optimize()
    pso_result = pso.optimize()

    assert np.isfinite(haoa_result["best_score"])
    assert np.isfinite(pso_result["best_score"])
    # A comparison harness must report both algorithms, not assume a winner.
    assert set(("best_score", "convergence_curve")).issubset(haoa_result)
    assert set(("best_score", "convergence_curve")).issubset(pso_result)
