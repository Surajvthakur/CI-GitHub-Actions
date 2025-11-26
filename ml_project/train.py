import os
from pathlib import Path

from .config import TrainingConfig
from .data import load_data, split_data
from .model import create_model, train_model, evaluate_model


def main() -> float:
    config = TrainingConfig()
    data_path = os.getenv("DATA_PATH", "data/sample_data.csv")
    output_dir = Path(os.getenv("OUTPUT_DIR", "artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df, config)

    model = create_model()
    model = train_model(model, X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)

    # Save a tiny “report”
    report_path = output_dir / "metrics.txt"
    with report_path.open("w") as f:
        f.write(f"accuracy={accuracy:.4f}\n")

    # In real project you’d serialize the model too
    return accuracy


if __name__ == "__main__":
    acc = main()
    print(f"Final accuracy: {acc:.4f}")
