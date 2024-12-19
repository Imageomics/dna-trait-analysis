from dataclasses import dataclass


@dataclass
class TrainingConfigs:
    train_ratio: float
    validation_ratio: float
    seed: int

    def __post_init__(self):
        pass
