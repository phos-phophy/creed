import argparse
import json
from pathlib import Path

from src.datasets import get_converter
from src.trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, metavar="<path to config>", required=True)
    parser.add_argument("-l", "--size_limit", type=str, metavar="<document number limit>", required=True)

    arguments = parser.parse_args()
    config_path = arguments.config_path
    size_limit = arguments.size_limit

    with Path(config_path).open('r') as file:
        config: dict = json.load(file)

    converter_config = config.pop("converter")
    train_path = config.pop("train_path")
    _ = config.pop("dev_path", None)

    converter = get_converter(**converter_config)

    train_documents = list(converter.convert(Path(train_path)))

    trainer = Trainer(config)

    total_len = len(train_documents)

    for i in total_len[::size_limit]:
        print(f"Trained on {i} examples out of {total_len}")
        train_docs = train_documents[i: i + size_limit]

        train_dataset = trainer.model.prepare_dataset(train_docs, True, False)
        trainer.train_model(train_dataset, rewrite=True)
