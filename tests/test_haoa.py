import numpy as np
import pytest

from HAOA.algorithm.haoa import HAOA


def sphere(x):
    return float(np.sum(np.asarray(x) ** 2))


def make_optimizer(**overrides):
    options = dict(
        objective_function=sphere,
        dimension=4,
        lower_bound=-3.0,
        upper_bound=3.0,
        population_size=12,
        max_iterations=15,
        random_state=1234,
    )
    options.update(overrides)
    return HAOA(**options)


def test_optimizer_returns_complete_finite_result():
    result = make_optimizer().optimize()
    assert np.isfinite(result["best_score"])
    assert result["best_solution"].shape == (4,)
    assert len(result["convergence_curve"]) == 15
    assert np.all(np.isfinite(result["convergence_curve"]))
    assert np.all(np.diff(result["convergence_curve"]) <= 1e-12)


def test_same_seed_reproduces_same_result():
    first = make_optimizer().optimize()
    second = make_optimizer().optimize()
    assert first["best_score"] == pytest.approx(second["best_score"])
    np.testing.assert_allclose(first["best_solution"], second["best_solution"])
    np.testing.assert_allclose(first["convergence_curve"], second["convergence_curve"])


def test_vector_bounds_are_respected():
    lower = np.array([-1.0, -2.0, -3.0, -4.0])
    upper = np.array([1.0, 2.0, 3.0, 4.0])
    optimizer = make_optimizer(lower_bound=lower, upper_bound=upper)
    result = optimizer.optimize()
    assert np.all(result["best_solution"] >= lower)
    assert np.all(result["best_solution"] <= upper)
    assert np.all(optimizer.population >= lower)
    assert np.all(optimizer.population <= upper)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dimension": 0},
        {"population_size": 1},
        {"max_iterations": 0},
        {"lower_bound": 2, "upper_bound": 1},
    ],
)
def test_invalid_configuration_fails_early(kwargs):
    with pytest.raises(ValueError):
        make_optimizer(**kwargs)


def test_non_finite_objective_fails_explicitly():
    optimizer = make_optimizer(objective_function=lambda _: float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        optimizer.optimize()
