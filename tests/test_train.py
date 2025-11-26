from ml_project.train import main


def test_main_runs_and_returns_accuracy(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "artifacts"))
    acc = main()
    assert 0.0 <= acc <= 1.0
