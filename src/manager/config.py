from typing import NamedTuple

from src.abstract import DiversifierConfig

from .manager import InitConfig, TrainingConfig


class MainConfig(NamedTuple):
    loader_config: dict
    init_config: dict  # InitConfig

    seed: int = 42
    save_path: str = None
    training_config: dict = {}  # TrainingConfig

    train_dataset_path: str = None
    dev_dataset_path: str = None
    eval_dataset_path: str = None
    test_dataset_path: str = None
    pred_dataset_path: str = None

    output_eval_path: str = None
    output_test_path: str = None
    output_pred_path: str = None

    train_diversifier: dict = {}  # DiversifierConfig
    dev_diversifier: dict = {}  # DiversifierConfig
    eval_diversifier: dict = {}  # DiversifierConfig
    test_diversifier: dict = {}  # DiversifierConfig
    pred_diversifier: dict = {}  # DiversifierConfig

    def init(self):
        self.init_config: InitConfig = InitConfig(**self.init_config)
        self.training_config: TrainingConfig = TrainingConfig(**self.training_config)

        self.train_diversifier: DiversifierConfig = DiversifierConfig(**self.train_diversifier)
        self.dev_diversifier: DiversifierConfig = DiversifierConfig(**self.dev_diversifier)
        self.eval_diversifier: DiversifierConfig = DiversifierConfig(**self.eval_diversifier)
        self.test_diversifier: DiversifierConfig = DiversifierConfig(**self.test_diversifier)
        self.pred_diversifier: DiversifierConfig = DiversifierConfig(**self.pred_diversifier)

        return self
