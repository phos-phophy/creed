import json
from pathlib import Path

from .datasets import get_converter
from .trainer import Trainer


def run_train(config_path: str):

    with Path(config_path).open('r') as file:
        config: dict = json.load(file)

    converter_config = config.pop("converter")
    train_path = Path(config.pop("train_path"))
    dev_path = Path(config.pop("dev_path"))

    converter = get_converter(converter_config)

    train_documents = list(converter.convert(train_path))
    dev_documents = list(converter.convert(dev_path))

    trainer = Trainer(config)

    train_dataset = trainer.model.prepare_dataset(train_documents, True, False)
    dev_dataset = trainer.model.prepare_dataset(dev_documents, True, True)

    trainer.train_model(train_dataset, dev_dataset)
