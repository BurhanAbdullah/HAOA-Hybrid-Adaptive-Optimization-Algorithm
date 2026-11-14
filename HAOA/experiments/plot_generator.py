import os
from pathlib import Path

import matplotlib.pyplot as plt


class PlotGenerator:
    """Generate plots for one complete HAOA benchmark run."""

    def __init__(self, output_directory=None):
        package_root = Path(__file__).resolve().parents[1]
        self.output_directory = Path(
            output_directory or package_root / "results" / "figures"
        )
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def generate_convergence_plot(self, all_results):
        fig, axis = plt.subplots(figsize=(10, 6))
        for result in all_results:
            axis.plot(
                result["convergence_curve"],
                label=result["benchmark"],
            )
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Best score")
        axis.set_title("HAOA convergence curves")
        if all_results:
            axis.legend()
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        file_path = self.output_directory / "convergence_plot.png"
        fig.savefig(file_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {file_path}")
        return file_path

    def generate_score_plot(self, all_results):
        fig, axis = plt.subplots(figsize=(10, 6))
        names = [result["benchmark"] for result in all_results]
        scores = [result["best_score"] for result in all_results]
        axis.bar(names, scores)
        axis.set_xlabel("Benchmark")
        axis.set_ylabel("Best score")
        axis.set_title("HAOA benchmark performance")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        file_path = self.output_directory / "score_plot.png"
        fig.savefig(file_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {file_path}")
        return file_path
