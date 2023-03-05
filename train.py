import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from src.abstract import DiversifierConfig
from src.loader import get_loader
from src.manager import InitConfig, ModelManager, TrainingConfig
from tqdm import tqdm

"""Config structure:
{
    "seed": int (optional),
    "loader_config": {...},
    "init_config": InitConfig,
    "training_config": TrainingConfig,
    "save_path": str,
    "train_dataset_path": str,
    "dev_dataset_path": str (optional),
    "test_dataset_path": str (optional),
    "output_eval_path": str (optional),
    "output_pred_path": str (optional),
    "train_diversifier": dict (optional),
    "dev_diversifier": dict (optional),
    "test_diversifier": dict (optional)
}
"""


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':

    # parse arguments and load main config
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, metavar="<path to config>", required=True)

    arguments = parser.parse_args()
    config_path = arguments.config_path

    with Path(config_path).open('r') as file:
        config: dict = json.load(file)

    # set seed
    set_seed(config.get("seed", 42))

    # parse main config
    loader_config = config["loader_config"]
    init_config = InitConfig(**config["init_config"])
    training_config = TrainingConfig(**config["training_config"])
    save_path = config["save_path"]
    train_dataset_path = config["train_dataset_path"]
    dev_dataset_path = config.get("dev_dataset_path", None)
    test_dataset_path = config.get("test_dataset_path", None)
    output_eval_path = config.get("output_eval_path", None)
    output_pred_path = config.get("output_pred_path", None)
    train_diversifier = DiversifierConfig(**config.get("train_diversifier", {}))
    dev_diversifier = DiversifierConfig(**config.get("dev_diversifier", {}))
    test_diversifier = DiversifierConfig(**config.get("test_diversifier", {}))

    # get documents
    print('Load the training and dev datasets')
    loader = get_loader(**loader_config)
    train_documents = list(tqdm(loader.load(Path(train_dataset_path)), desc='Training documents'))
    dev_documents = list(tqdm(loader.load(Path(dev_dataset_path)), desc='Dev documents')) if dev_dataset_path else None

    # train, evaluate, predict and save
    print('Init the model and its manager')
    manager = ModelManager(init_config)

    print('Start training')
    manager.train(training_config, train_diversifier, dev_diversifier, train_documents, dev_documents)

    print(f'Save the model in the file {Path(save_path)}')
    manager.save(Path(save_path), False)

    batch_size = training_config.training_arguments.get("per_device_eval_batch_size", 5)

    if dev_documents and output_eval_path:
        print(f'Evaluate the model. The results will be saved in the file {Path(output_eval_path)}')
        manager.evaluate(dev_documents, dev_diversifier, Path(output_eval_path), batch_size)

    if test_dataset_path and output_pred_path:
        print(f'Load the test dataset and make predictions that will be saved in the file {Path(output_pred_path)}')
        test_documents = list(tqdm(loader.load(Path(test_dataset_path)), desc='Test documents'))
        manager.predict(test_documents, test_diversifier, Path(output_pred_path), batch_size)

    print(f'Save the model in the file {Path(save_path)}')
    manager.save(Path(save_path), True)
