from functools import partial
from pathlib import Path
from typing import List, NamedTuple

import torch
from src.abstract import AbstractWrapperModel, DiversifierConfig, Document
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
            train_diversifier: DiversifierConfig,
            dev_diversifier: DiversifierConfig,
            train_documents: List[Document],
            dev_documents: List[Document] = None,
    ):
        train_params = TrainingArguments(**config.training_arguments)
        compute_metrics = config.compute_metrics

        train_dataset = self.model.prepare_dataset(train_documents, train_diversifier, 'Prepare training dataset', True, False)

        dev_dataset = None
        if dev_documents:
            dev_dataset = self.model.prepare_dataset(dev_documents, dev_diversifier, 'Prepare dev dataset', True, True).prepare_documents()

        compute_metrics = partial(score_model, relations=self.model.relations) if compute_metrics else None

        torch.cuda.empty_cache()

        trainer = Trainer(
            model=self.model,
            args=train_params,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=collate_fn,
            compute_metrics=compute_metrics
        )

        trainer.train()

    def evaluate(self, documents: List[Document], diversifier: DiversifierConfig, output_path: Path = None, batch_size: int = 5):
        dataset = self.model.prepare_dataset(documents, diversifier, 'Prepare dev dataset', True, True).prepare_documents()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        torch.cuda.empty_cache()
        self.model.evaluate(dataloader, output_path)

    def predict(self, documents: List[Document], diversifier: DiversifierConfig, output_path: Path, batch_size: int = 5):
        dataset = self.model.prepare_dataset(documents, diversifier, 'Prepare pred dataset', False, True).prepare_documents()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        torch.cuda.empty_cache()
        self.model.predict(documents, dataloader, output_path)

    def test(self, documents: List[Document], diversifier: DiversifierConfig, output_path: Path, batch_size: int = 5):
        dataset = self.model.prepare_dataset(documents, diversifier, 'Prepare test dataset', True, True).prepare_documents()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        torch.cuda.empty_cache()
        self.model.test(dataloader, output_path)

    def save(self, save_path: Path, rewrite: bool = False):
        self.model.save(path=save_path, rewrite=rewrite)
