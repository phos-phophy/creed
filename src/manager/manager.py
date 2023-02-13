from functools import partial
from pathlib import Path
from typing import List, NamedTuple

import torch
from src.abstract import AbstractDataset, AbstractModel, Document
from src.models import get_model
from transformers import Trainer, TrainingArguments

from .collate import collate_fn
from .score import score_model


# See https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/trainer#transformers.TrainingArguments
class TrainingConfig(NamedTuple):
    training_arguments: dict
    save_path: str
    compute_metrics: bool = True


class ModelManager:
    def __init__(self, config):
        """
        Config's structure:
        {
            "load_path": str (optional if other parameters are not specified)
            ...
        }
        """

        load_path = config.get("load_path", None)
        model = AbstractModel.load(Path(load_path)) if load_path else get_model(**config)
        self.model = model.cuda() if torch.cuda.is_available() else model

    def train(
            self,
            config: TrainingConfig,
            train_documents: List[Document],
            dev_documents: List[Document] = None,
            rewrite: bool = False
    ):

        save_path = Path(config.save_path)
        train_params = TrainingArguments(**config.training_arguments)
        compute_metrics = config.compute_metrics

        train_dataset: AbstractDataset = self.model.prepare_dataset(train_documents, True, False)
        dev_dataset: AbstractDataset = self.model.prepare_dataset(dev_documents, True, True) if dev_documents else None

        compute_metrics = partial(score_model, relations=self.model.relations) if compute_metrics else None

        trainer = Trainer(
            model=self.model,
            args=train_params,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=collate_fn,
            compute_metrics=compute_metrics
        )

        trainer.train()

        self.model.save(path=save_path, rewrite=rewrite)

    def evaluate(self, documents: List[Document], output_path: str = None):
        dataset: AbstractDataset = self.model.prepare_dataset(documents, True, True)
        self.model.evaluate(dataset, output_path)

    def predict(self, documents: List[Document], output_path: str = None):
        dataset: AbstractDataset = self.model.prepare_dataset(documents, False, False)
        self.model.predict(dataset, output_path)
