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
    "train_dataset_path": str
    "dev_dataset_path": str (optional),
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

    with Path(arguments.config_path).open('r') as file:
        config: dict = json.load(file)

    # parse main config
    loader_config = config["loader_config"]
    init_config = InitConfig(**config["init_config"])
    training_config = TrainingConfig(**config["training_config"])
    train_dataset_path = config["train_dataset_path"]
    dev_dataset_path = config.get("dev_dataset_path", None)
    output_eval_path = config_path.get("output_eval_path", None)
    output_pred_path = config_path.get("output_pred_path", None)

    # get documents
    loader = get_loader(**loader_config)
    train_documents = list(loader.load(Path(train_dataset_path)))
    dev_documents = list(loader.load(Path(dev_dataset_path))) if dev_dataset_path else None

    # train, evaluate, predict and save
    manager = ModelManager(init_config)
    manager.train(training_config, train_documents, dev_documents)

    if dev_documents and output_eval_path:
        manager.evaluate(dev_documents, output_eval_path, training_config.training_arguments.get("per_device_eval_batch_size", 5))

    if output_pred_path and output_pred_path:
        manager.predict(dev_documents, output_pred_path, training_config.training_arguments.get("per_device_eval_batch_size", 5))
