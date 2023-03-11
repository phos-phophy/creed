import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from src.loader import get_loader
from src.manager import MainConfig, ModelManager
from tqdm import tqdm


def load_config():
    """ Parse arguments and load main config """

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, metavar="<path to config>", required=True)

    arguments = parser.parse_args()
    config_path = arguments.config_path

    with Path(config_path).open('r') as file:
        config: dict = json.load(file)

    return MainConfig(**config)


def set_seed(config: MainConfig):
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)


def main():
    config: MainConfig = load_config()
    set_seed(config)

    print('Init the model, its manager and loader')
    manager = ModelManager(config.init_config)
    loader = get_loader(**config.loader_config)

    # train model
    if config.train_dataset_path and config.save_path:
        print('Load the training and dev datasets')
        train_documents = list(tqdm(loader.load(Path(config.train_dataset_path)), desc='Training documents'))
        dev_documents = list(tqdm(loader.load(Path(config.dev_dataset_path)), desc='Dev documents')) if config.dev_dataset_path else None

        print('Start training')
        manager.train(config.training_config, config.train_diversifier, config.dev_diversifier, train_documents, dev_documents)

        print(f'Save the model in the file {Path(config.save_path)}')
        manager.save(Path(config.save_path), False)

    batch_size = config.training_config.training_arguments.get("per_device_eval_batch_size", 5)

    # evaluate the model
    if config.eval_dataset_path and config.output_eval_path and config.save_path:
        print(f'Load the eval dataset and evaluate the model. The results will be saved in the file {Path(config.output_eval_path)}')
        eval_documents = list(tqdm(loader.load(Path(config.eval_dataset_path)), desc='Eval documents'))
        manager.evaluate(eval_documents, config.eval_diversifier, Path(config.output_eval_path), batch_size)

        print(f'Resave the model in the file {Path(config.save_path)}')
        manager.save(Path(config.save_path), True)

    # test the model on the public test dataset
    if config.test_dataset_path and config.output_test_path:
        print(f'Load the test dataset and test the model. The results will be saved in the file {Path(config.output_test_path)}')
        test_documents = list(tqdm(loader.load(Path(config.test_dataset_path)), desc='Test documents'))
        manager.test(test_documents, config.test_diversifier, Path(config.output_test_path), batch_size)

    # predict on the private test dataset
    if config.pred_dataset_path and config.output_pred_path:
        print(f'Load the pred dataset and make predictions that will be saved in the file {Path(config.output_pred_path)}')
        pred_documents = list(tqdm(loader.load(Path(config.pred_dataset_path)), desc='Test documents'))
        manager.predict(pred_documents, config.pred_diversifier, Path(config.output_pred_path), batch_size)


if __name__ == '__main__':
    main()
