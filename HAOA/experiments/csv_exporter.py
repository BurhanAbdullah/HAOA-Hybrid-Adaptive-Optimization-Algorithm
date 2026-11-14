import csv
import os
from pathlib import Path


class CSVExporter:
    """Write benchmark summaries and convergence histories."""

    def __init__(self, output_directory=None):
        # Keep outputs inside the package by default. This works regardless
        # of the caller's current working directory.
        package_root = Path(__file__).resolve().parents[1]
        self.output_directory = Path(
            output_directory or package_root / "results" / "csv"
        )
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def export_summary(self, all_results):
        file_path = self.output_directory / "summary.csv"
        with file_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["benchmark", "best_score", "execution_time"])
            for result in all_results:
                writer.writerow([
                    result["benchmark"],
                    result["best_score"],
                    result["execution_time"],
                ])
        print(f"Saved: {file_path}")
        return file_path

    def export_convergence(self, all_results):
        file_path = self.output_directory / "convergence.csv"
        with file_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["benchmark", "iteration", "best_score"])
            for result in all_results:
                for iteration, score in enumerate(
                    result["convergence_curve"], start=1
                ):
                    writer.writerow([result["benchmark"], iteration, score])
        print(f"Saved: {file_path}")
        return file_path
