"""Hybrid Adaptive Optimization Algorithm (HAOA)."""

import math

import numpy as np


class HAOA:
    """Bounded population-based minimizer.

    The objective must accept a one-dimensional NumPy vector and return one
    finite scalar. Bounds may be scalars or one value per dimension.
    """

    def __init__(
        self,
        objective_function,
        dimension,
        lower_bound,
        upper_bound,
        population_size,
        max_iterations,
        *,
        random_state=None,
        verbose=False,
    ):
        if not callable(objective_function):
            raise TypeError("objective_function must be callable")
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        if not isinstance(population_size, int) or isinstance(population_size, bool) or population_size < 2:
            raise ValueError("population_size must be an integer >= 2")
        if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")

        self.objective_function = objective_function
        self.dimension = dimension
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.verbose = bool(verbose)
        self.rng = np.random.default_rng(random_state)

        self.lower_bound = np.broadcast_to(
            np.asarray(lower_bound, dtype=float), (dimension,)
        ).copy()
        self.upper_bound = np.broadcast_to(
            np.asarray(upper_bound, dtype=float), (dimension,)
        ).copy()
        if not np.all(np.isfinite(self.lower_bound)) or not np.all(np.isfinite(self.upper_bound)):
            raise ValueError("bounds must be finite")
        if np.any(self.lower_bound >= self.upper_bound):
            raise ValueError("each lower bound must be below its upper bound")

        self.population = self.rng.uniform(
            self.lower_bound, self.upper_bound, (population_size, dimension)
        )
        self.velocity = np.zeros_like(self.population)
        self.best_solution = None
        self.best_score = float("inf")
        self.convergence_curve = []
        self.stagnation_counter = 0
        self._levy_sigma = self._calculate_levy_sigma()

    @staticmethod
    def _calculate_levy_sigma(beta=1.5):
        numerator = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        denominator = (
            math.gamma((1 + beta) / 2)
            * beta
            * 2 ** ((beta - 1) / 2)
        )
        return (numerator / denominator) ** (1 / beta)

    def _evaluate(self, position):
        value = self.objective_function(position.copy())
        value_array = np.asarray(value)
        if value_array.ndim != 0:
            raise ValueError("objective_function must return a scalar")
        value = float(value_array)
        if not np.isfinite(value):
            raise ValueError("objective_function returned a non-finite value")
        return value

    def evaluate_population(self):
        improved = False
        for individual in self.population:
            fitness = self._evaluate(individual)
            if fitness < self.best_score:
                self.best_score = fitness
                self.best_solution = individual.copy()
                improved = True
        self.stagnation_counter = 0 if improved else self.stagnation_counter + 1

    def levy_flight(self):
        u = self.rng.normal(size=self.dimension) * self._levy_sigma
        v = self.rng.normal(size=self.dimension)
        return u / (np.abs(v) ** (1 / 1.5) + 1e-10)

    def update_population(self, iteration):
        if self.best_solution is None:
            raise RuntimeError("evaluate_population must run before update_population")

        progress = iteration / self.max_iterations
        inertia_weight = 0.95 - 0.6 * progress
        adaptive_learning = 1.8 * (1.0 - progress)
        mutation_strength = 0.003 * (1.0 - progress)
        span = self.upper_bound - self.lower_bound
        velocity_limit = 0.2 * span
        elite = self.best_solution.copy()

        for i in range(self.population_size):
            current = self.population[i]
            r1 = self.rng.random(self.dimension)
            r2 = self.rng.random(self.dimension)
            r3 = self.rng.random(self.dimension)

            direction = elite - current
            random_target = self.rng.uniform(
                self.lower_bound, self.upper_bound, self.dimension
            )
            exploration = adaptive_learning * r2 * (random_target - current)
            exploitation = 1.4 * adaptive_learning * r1 * direction
            elite_refinement = 0.15 * r3 * direction

            velocity = (
                inertia_weight * self.velocity[i]
                + exploitation
                + 0.15 * exploration
                + elite_refinement
            )
            self.velocity[i] = np.clip(velocity, -velocity_limit, velocity_limit)

            mutation = mutation_strength * span * self.rng.normal(size=self.dimension)
            candidate = current + self.velocity[i] + mutation

            if self.stagnation_counter > 25:
                candidate = candidate + 0.005 * span * self.levy_flight()

            if progress > 0.75:
                candidate = candidate + 0.03 * (elite - candidate)

            candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
            current_fitness = self._evaluate(current)
            candidate_fitness = self._evaluate(candidate)

            if candidate_fitness < current_fitness:
                self.population[i] = candidate
            else:
                # Do not retain a velocity that repeatedly pushes away from
                # a rejected point; damp it to keep the search stable.
                self.velocity[i] *= 0.5

    def optimize(self):
        """Run the search and return the best solution and convergence history."""
        for iteration in range(self.max_iterations):
            self.evaluate_population()
            self.update_population(iteration)
            self.convergence_curve.append(self.best_score)

            if self.verbose:
                print(
                    f"Iteration {iteration + 1}/{self.max_iterations}"
                    f" | Best score: {self.best_score:.10e}"
                )

        # Evaluate once more so the final population is represented in the
        # returned best-so-far state.
        self.evaluate_population()
        if self.convergence_curve:
            self.convergence_curve[-1] = min(
                self.convergence_curve[-1], self.best_score
            )
        return {
            "best_score": self.best_score,
            "best_solution": self.best_solution.copy(),
            "convergence_curve": list(self.convergence_curve),
        }
