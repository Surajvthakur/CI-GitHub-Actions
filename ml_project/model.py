from typing import Any

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def create_model() -> BaseEstimator:
    # Simple baseline model
    model = LogisticRegression(max_iter=200)
    return model


def train_model(model: BaseEstimator, X_train, y_train) -> BaseEstimator:
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: BaseEstimator, X_test, y_test) -> float:
    preds = model.predict(X_test)
    accuracy: float = float(accuracy_score(y_test, preds))
    if not (0.0 <= accuracy <= 1.0):
        raise ValueError("Accuracy out of expected range 0–1")
    return accuracy
