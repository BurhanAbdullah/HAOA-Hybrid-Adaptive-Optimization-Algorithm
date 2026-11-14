import csv

from HAOA.experiments.csv_exporter import CSVExporter


def test_csv_export_schema_and_rows(tmp_path):
    results = [{
        "benchmark": "Sphere",
        "best_score": 0.125,
        "execution_time": 1.5,
        "convergence_curve": [2.0, 1.0, 0.125],
    }]
    exporter = CSVExporter(output_directory=tmp_path)
    summary = exporter.export_summary(results)
    convergence = exporter.export_convergence(results)

    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["benchmark"] == "Sphere"
    assert float(rows[0]["best_score"]) == 0.125

    with convergence.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[-1]["iteration"] == "3"
    assert float(rows[-1]["best_score"]) == 0.125
