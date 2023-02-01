import argparse
import json
from pathlib import Path

from src.datasets import get_loader
from src.manager import ModelManager


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, metavar="<path to config>", required=True)

    arguments = parser.parse_args()
    config_path = arguments.config_path

    with Path(config_path).open('r') as file:
        config: dict = json.load(file)

    loader_config = config.pop("loader")
    train_path = config.pop("train_path")
    dev_path = config.pop("dev_path", None)

    loader = get_loader(**loader_config)

    train_documents = list(loader.load(Path(train_path)))
    dev_documents = list(loader.load(Path(dev_path))) if dev_path else None

    manager = ModelManager(config)
    manager.train_model(train_documents, dev_documents)
