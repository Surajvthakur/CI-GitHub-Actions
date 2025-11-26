from pathlib import Path

import pandas as pd

from ml_project.data import load_data


def test_load_data_has_target():
    df = load_data("data/sample_data.csv")
    assert "target" in df.columns


def test_load_data_no_missing_values(tmp_path: Path):
    df = load_data("data/sample_data.csv")
    assert df.isna().sum().sum() == 0

    # Negative test: create a file with missing values
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"f1": [1, None], "target": [0, 1]}).to_csv(bad_csv, index=False)
    try:
        load_data(str(bad_csv))
        assert False, "Expected ValueError for missing values"
    except ValueError:
        pass
