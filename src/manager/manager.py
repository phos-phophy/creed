from functools import partial
from pathlib import Path
from typing import List, NamedTuple

import torch
from src.abstract import AbstractDataset, AbstractWrapperModel, Document
from src.models import get_model
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

from .collate import collate_fn
from .score import score_model


class InitConfig(NamedTuple):
    load_path: str = None
    model_params: dict = {}  # load_path and model_params are mutually exclusive


# See https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/trainer#transformers.TrainingArguments
class TrainingConfig(NamedTuple):
    training_arguments: dict
    compute_metrics: bool = True


class ModelManager:
    def __init__(self, config: InitConfig):
        load_path = config.load_path
        model = AbstractWrapperModel.load(Path(load_path)) if load_path else get_model(**config.model_params)
        self.model = model.cuda() if torch.cuda.is_available() else model

    def train(
            self,
            config: TrainingConfig,
            train_documents: List[Document],
            dev_documents: List[Document] = None
    ):
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

    def evaluate(self, documents: List[Document], output_path: Path = None, batch_size: int = 5):
        dataset: AbstractDataset = self.model.prepare_dataset(documents, True, True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.model.evaluate(dataloader, output_path)

    def predict(self, documents: List[Document], output_path: Path, batch_size: int = 5):
        dataset: AbstractDataset = self.model.prepare_dataset(documents, False, True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.model.predict(documents, dataloader, output_path)

    def save(self, save_path: Path, rewrite: bool = False):
        self.model.save(path=save_path, rewrite=rewrite)
