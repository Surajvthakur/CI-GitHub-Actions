from ml_project.config import TrainingConfig
from ml_project.data import load_data, split_data
from ml_project.model import create_model, evaluate_model, train_model


def test_model_training_and_evaluation():
    cfg = TrainingConfig()
    df = load_data("data/sample_data.csv")
    X_train, X_test, y_train, y_test = split_data(df, cfg)

    model = create_model()
    model = train_model(model, X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)

    # with toy data, just ensure we get a valid number
    assert 0.0 <= accuracy <= 0.5
