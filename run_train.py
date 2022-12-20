import argparse
import json
from pathlib import Path

from src.datasets import get_converter
from src.trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, metavar="<path to config>", required=True)

    arguments = parser.parse_args()
    config_path = arguments.config_path

    with Path(config_path).open('r') as file:
        config: dict = json.load(file)

    converter_config = config.pop("converter")
    train_path = config.pop("train_path")
    dev_path = config.pop("dev_path", None)

    converter = get_converter(**converter_config)

    train_documents = list(converter.convert(Path(train_path)))
    dev_documents = list(converter.convert(Path(dev_path))) if dev_path else None

    trainer = Trainer(config)

    train_dataset = trainer.model.prepare_dataset(train_documents, True, False)
    dev_dataset = trainer.model.prepare_dataset(dev_documents, True, True) if dev_documents else None

    trainer.train_model(train_dataset, dev_dataset)
