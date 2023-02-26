import argparse
import json
from pathlib import Path

from src.loader import get_loader
from src.manager import InitConfig, ModelManager, TrainingConfig

"""Config structure:
{
    "loader_config": {...},
    "init_config": InitConfig,
    "training_config": TrainingConfig,
    "save_path": str,
    "train_dataset_path": str,
    "dev_dataset_path": str (optional),
    "test_dataset_path": str (optional),
    "output_eval_path": str (optional),
    "output_pred_path": str (optional)
}
"""

if __name__ == '__main__':

    # parse arguments and load main config
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, metavar="<path to config>", required=True)

    arguments = parser.parse_args()
    config_path = arguments.config_path

    with Path(config_path).open('r') as file:
        config: dict = json.load(file)

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

    # get documents
    loader = get_loader(**loader_config)
    train_documents = list(loader.load(Path(train_dataset_path)))
    dev_documents = list(loader.load(Path(dev_dataset_path))) if dev_dataset_path else None

    # train, evaluate, predict and save
    manager = ModelManager(init_config)
    manager.train(training_config, train_documents, dev_documents)

    manager.save(Path(save_path), False)

    batch_size = training_config.training_arguments.get("per_device_eval_batch_size", 5)

    if dev_documents and output_eval_path:
        manager.evaluate(dev_documents, Path(output_eval_path), batch_size)

    if test_dataset_path and output_pred_path:
        test_documents = list(loader.load(Path(test_dataset_path)))
        manager.predict(test_documents, Path(output_pred_path), batch_size)

    manager.save(Path(save_path), True)
