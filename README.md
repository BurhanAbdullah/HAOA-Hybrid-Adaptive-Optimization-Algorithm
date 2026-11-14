# HAOA — Hybrid Adaptive Optimization Algorithm

HAOA is a research-oriented, bounded, population-based minimization algorithm. This repository includes the HAOA implementation, standard numerical benchmark functions, a PSO baseline, experiment runners, and CSV/plot exporters.

## Features

- Hybrid exploration and elite-directed exploitation
- Adaptive velocity, mutation, and stagnation-triggered Lévy exploration
- Scalar and per-dimension bounds
- Reproducible runs through an explicit random seed
- Sphere, Rastrigin, Rosenbrock, Ackley, and Griewank benchmarks
- HAOA/PSO comparison and experiment output in CSV/PNG formats

## Repository layout

```text
HAOA/
├── algorithm/haoa.py
├── benchmark_functions/
├── comparison_algorithms/pso.py
├── experiments/
│   ├── benchmark_manager.py
│   ├── csv_exporter.py
│   ├── dataset_comparison.py
│   └── full_runner.py
├── results/
└── visualization/
```

## Requirements

Python 3.9 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the complete benchmark suite

Run from the repository root:

```bash
python -m HAOA.experiments.full_runner
```

The runner evaluates the five benchmark functions and writes:
- `HAOA/results/csv/summary.csv`
- `HAOA/results/csv/convergence.csv`
- `HAOA/results/figures/convergence_plot.png`
- `HAOA/results/figures/score_plot.png`

To compare HAOA with the included PSO implementation:

```bash
python -m HAOA.experiments.dataset_comparison
```

## Use HAOA directly

```python
import numpy as np
from HAOA.algorithm.haoa import HAOA

def sphere(x):
    return np.sum(x ** 2)

optimizer = HAOA(
    objective_function=sphere,
    dimension=10,
    lower_bound=-5.12,
    upper_bound=5.12,
    population_size=40,
    max_iterations=200,
    random_state=42,
)
result = optimizer.optimize()
print(result["best_score"])
print(result["best_solution"])
```

The objective function must accept a one-dimensional NumPy array and return a finite scalar. The optimizer minimizes the objective and enforces the supplied bounds.

## Tests

Install the requirements and run:

```bash
python -m pytest
```

## Research note

Benchmark scores depend on the random seed, search budget, dimensionality, and bounds. The repository does not treat historical sample numbers as guaranteed outcomes; report aggregate results over multiple independent seeds when making comparative research claims.
