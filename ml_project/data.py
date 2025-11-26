from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import TrainingConfig


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Simple data validation
    if "target" not in df.columns:
        raise ValueError("Data must contain 'target' column")
    if df.isna().sum().sum() > 0:
        raise ValueError("Data contains missing values")
    return df


def split_data(df: pd.DataFrame, config: TrainingConfig): #-> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[config.target_column])
    y = df[config.target_column]
    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )
