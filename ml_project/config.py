from dataclasses import dataclass


@dataclass
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    target_column: str = "target"
