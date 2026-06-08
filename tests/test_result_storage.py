import pandas as pd

from src.common import append_result


def test_result_storage_handles_different_metric_columns(tmp_path):
    result_path = tmp_path / "results.csv"
    append_result(result_path, {"model": "a", "recall": 0.1})
    append_result(result_path, {"model": "b", "recall": 0.2, "coverage": 0.3})
    frame = pd.read_csv(result_path)
    assert len(frame) == 2
    assert set(frame.columns) >= {"run_id", "model", "recall", "coverage"}
    assert frame["schema_version"].nunique() == 1
