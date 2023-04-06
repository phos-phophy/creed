from copy import deepcopy
from typing import NamedTuple

from src.abstract import DiversifierConfig


class ModelInitConfig(NamedTuple):
    load_path: str = None
    model_params: dict = {}  # load_path and model_params are mutually exclusive


# See https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/trainer#transformers.TrainingArguments
class TrainingConfig(NamedTuple):
    training_arguments: dict
    compute_metrics: bool = True


class ManagerConfig(NamedTuple):
    loader_config: dict
    model_init_config: ModelInitConfig

    seed: int = 42
    save_path: str = None
    cache_dir: str = None
    training_config: TrainingConfig = TrainingConfig({}, False)

    train_dataset_path: str = None
    dev_dataset_path: str = None
    eval_dataset_path: str = None
    test_dataset_path: str = None
    pred_dataset_path: str = None

    output_eval_path: str = None
    output_test_path: str = None
    output_pred_path: str = None

    train_diversifier: DiversifierConfig = DiversifierConfig()
    dev_diversifier: DiversifierConfig = DiversifierConfig()
    eval_diversifier: DiversifierConfig = DiversifierConfig()
    test_diversifier: DiversifierConfig = DiversifierConfig()
    pred_diversifier: DiversifierConfig = DiversifierConfig()

    @classmethod
    def from_dict(cls, config: dict):
        config = deepcopy(config)

        config['model_init_config'] = ModelInitConfig(**config['model_init_config'])
        config['training_config'] = TrainingConfig(**config['training_config'])

        config['train_diversifier'] = DiversifierConfig(**config.get('train_diversifier', {}))
        config['dev_diversifier'] = DiversifierConfig(**config.get('dev_diversifier', {}))
        config['eval_diversifier'] = DiversifierConfig(**config.get('eval_diversifier', {}))
        config['test_diversifier'] = DiversifierConfig(**config.get('test_diversifier', {}))
        config['pred_diversifier'] = DiversifierConfig(**config.get('pred_diversifier', {}))

        return cls(**config)
