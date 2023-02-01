from functools import partial
from pathlib import Path
from typing import List

import torch
from src.abstract import AbstractDataset, AbstractModel, Document
from src.models import get_model
from transformers import Trainer, TrainingArguments

from .collate import collate_fn
from .score import score_model


class ModelManager:
    def __init__(self, config):
        """
        Config's structure:
        {
            "training_arguments": {
                ...
                See https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/trainer#transformers.TrainingArguments
                ...
            },

            "save_path": str,

            "model": {
                "load_path": str (optional if other parameters are not specified)
                ...
            }
        }
        """

        self.save_path = Path(config["save_path"])
        self.train_params = TrainingArguments(**config["training_arguments"])
        self.model_config = config["model"]

    def init_model(self) -> AbstractModel:
        load_path = self.model_config.get("load_path", None)
        model = AbstractModel.load(Path(load_path)) if load_path else get_model(**self.model_config)

        return model.cuda() if torch.cuda.is_available() else model

    def train_model(self, train_documents: List[Document], dev_documents: List[Document] = None, rewrite: bool = False):

        model = self.init_model()

        train_dataset: AbstractDataset = model.prepare_dataset(train_documents, True, False)
        dev_dataset: AbstractDataset = model.prepare_dataset(dev_documents, True, True) if dev_documents else None

        trainer = Trainer(
            model=model,
            args=self.train_params,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=collate_fn,
            compute_metrics=partial(score_model, relations=model.relations)
        )

        trainer.train()

        model.save(path=self.save_path, rewrite=rewrite)
        return model
