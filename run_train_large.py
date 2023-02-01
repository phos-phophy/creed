import argparse
import json
from pathlib import Path

import numpy as np
from src.loader import get_loader
from src.manager import ModelManager


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, metavar="<path to config>", required=True)
    parser.add_argument("-l", "--size_limit", type=int, metavar="<document number limit>", required=True)

    arguments = parser.parse_args()
    config_path = arguments.config_path
    size_limit = arguments.size_limit

    with Path(config_path).open('r') as file:
        config: dict = json.load(file)

    loader_config = config.pop("loader")
    train_path = config.pop("train_path")
    _ = config.pop("dev_path", None)

    loader = get_loader(**loader_config)

    train_documents = list(loader.load(Path(train_path)))

    manager = ModelManager(config)

    total_len = len(train_documents)

    for i in np.arange(total_len, step=size_limit):
        print(f"Trained on {i} examples out of {total_len}")
        train_docs = train_documents[i: i + size_limit]
        manager.train_model(train_docs, rewrite=True)
