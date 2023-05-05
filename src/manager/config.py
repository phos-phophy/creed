from copy import deepcopy
from pathlib import Path
from typing import NamedTuple

from src.abstract import DiversifierConfig


class ModelInitConfig(NamedTuple):
    load_path: Path = None
    model_params: dict = {}  # load_path and model_params are mutually exclusive

    @classmethod
    def from_dict(cls, config: dict):
        config['load_path'] = Path(config['load_path']) if 'load_path' in config else None
        return cls(**config)


# See https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/trainer#transformers.TrainingArguments
class TrainingConfig(NamedTuple):
    training_arguments: dict
    compute_metrics: bool = True

    @classmethod
    def from_dict(cls, config: dict, output_dir: str):
        config['training_arguments']['logging_dir'] = str(Path(output_dir) / 'log_dir')
        config['training_arguments']['output_dir'] = str(Path(output_dir) / 'output')
        return cls(**config)


class ManagerConfig(NamedTuple):
    loader_config: dict
    model_init_config: ModelInitConfig

    save_path: Path
    output_eval_path: Path
    output_test_path: Path
    output_pred_path: Path

    seed: int = 42
    training_config: TrainingConfig = TrainingConfig({}, False)
    extra_training_config: dict = {}

    train_dataset_path: Path = None
    dev_dataset_path: Path = None
    eval_dataset_path: Path = None
    test_dataset_path: Path = None
    pred_dataset_path: Path = None

    train_diversifier: DiversifierConfig = DiversifierConfig()
    dev_diversifier: DiversifierConfig = DiversifierConfig()
    eval_diversifier: DiversifierConfig = DiversifierConfig()
    test_diversifier: DiversifierConfig = DiversifierConfig()
    pred_diversifier: DiversifierConfig = DiversifierConfig()

    @classmethod
    def from_dict(cls, config: dict, output_dir: str):
        config = deepcopy(config)

        config['save_path'] = Path(output_dir) / 'model.pkl'
        config['output_eval_path'] = Path(output_dir) / 'eval_results.json'
        config['output_test_path'] = Path(output_dir) / 'test_results.json'
        config['output_pred_path'] = Path(output_dir) / 'result.json'

        config['model_init_config'] = ModelInitConfig.from_dict(config['model_init_config'])
        config['training_config'] = TrainingConfig.from_dict(config['training_config'], output_dir)

        config['train_dataset_path'] = Path(config['train_dataset_path']) if 'train_dataset_path' in config else None
        config['dev_dataset_path'] = Path(config['dev_dataset_path']) if 'dev_dataset_path' in config else None
        config['eval_dataset_path'] = Path(config['eval_dataset_path']) if 'eval_dataset_path' in config else None
        config['test_dataset_path'] = Path(config['test_dataset_path']) if 'test_dataset_path' in config else None
        config['pred_dataset_path'] = Path(config['pred_dataset_path']) if 'pred_dataset_path' in config else None

        config['train_diversifier'] = DiversifierConfig(**config.get('train_diversifier', {}))
        config['dev_diversifier'] = DiversifierConfig(**config.get('dev_diversifier', {}))
        config['eval_diversifier'] = DiversifierConfig(**config.get('eval_diversifier', {}))
        config['test_diversifier'] = DiversifierConfig(**config.get('test_diversifier', {}))
        config['pred_diversifier'] = DiversifierConfig(**config.get('pred_diversifier', {}))

        return cls(**config)
