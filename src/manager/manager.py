import random
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
import torch
from src.abstract import AbstractModel, Document
from src.loader import get_loader
from src.models import get_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Trainer, TrainingArguments

from .config import ManagerConfig
from .score import score_model


class ModelManager:
    def __init__(self, config: ManagerConfig):
        """ Init the model, its manager and loader """

        self.config = config
        self.loader = get_loader(**config.loader_config)

        self.set_seed()

        load_path = config.model_init_config.load_path
        model = AbstractModel.load(load_path) if load_path else get_model(**config.model_init_config.model_params)
        self.model = model.cuda() if torch.cuda.is_available() else model

    def train(self):

        if not self.config.train_dataset_path:
            return

        print('Load the training and dev datasets')
        train_documents = self.load_dataset(self.config.train_dataset_path, 'Training documents')
        dev_documents = self.load_dataset(self.config.dev_dataset_path, 'Dev documents') if self.config.dev_dataset_path else None

        train_desc = 'Prepare training dataset'
        dev_desc = 'Prepare dev dataset'

        train_diversifier = self.config.train_diversifier
        dev_diversifier = self.config.dev_diversifier

        train_dataset = self.model.prepare_dataset(train_documents, train_diversifier, train_desc, True, False)
        dev_dataset = self.model.prepare_dataset(dev_documents, dev_diversifier, dev_desc, True, True) if dev_documents else None

        dev_dataset = dev_dataset.prepare_documents() if dev_dataset else None

        train_params = TrainingArguments(**self.config.training_config.training_arguments)
        compute_metrics = partial(score_model, relations=self.model.relations) if self.config.training_config.compute_metrics else None

        torch.cuda.empty_cache()

        trainer = Trainer(
            model=self.model,
            args=train_params,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=self.model.collate_fn,
            compute_metrics=compute_metrics,
            optimizers=self.model.create_optimizers(self.config.extra_training_config)
        )

        print('Start training')
        trainer.train()

        self.save_model(rewrite=False)

    def evaluate(self):
        """ Evaluate the model """

        if not self.config.eval_dataset_path:
            return

        print(f'Load the eval dataset and evaluate the model. The results will be saved in the file {self.config.output_eval_path}')
        documents = self.load_dataset(self.config.eval_dataset_path, 'Eval documents')

        diversifier = self.config.eval_diversifier
        batch_size = self.config.training_config.training_arguments.get("per_device_eval_batch_size", 5)

        dataset = self.model.prepare_dataset(documents, diversifier, 'Prepare eval dataset', True, True).prepare_documents()

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self.model.collate_fn)

        torch.cuda.empty_cache()
        self.model.evaluate(dataloader, self.config.output_eval_path)

        self.save_model(rewrite=True)

    def test(self):
        """ Test the model on the public test dataset """

        if not self.config.test_dataset_path:
            return

        print(f'Load the test dataset and test the model. The results will be saved in the file {self.config.output_test_path}')
        documents = self.load_dataset(self.config.test_dataset_path, 'Test_documents')

        diversifier = self.config.test_diversifier
        batch_size = self.config.training_config.training_arguments.get("per_device_eval_batch_size", 5)

        dataset = self.model.prepare_dataset(documents, diversifier, 'Prepare test dataset', True, True).prepare_documents()

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self.model.collate_fn)

        torch.cuda.empty_cache()
        self.model.test(dataloader, self.config.output_test_path)

    def predict(self):
        """ Predict on the private test dataset """

        if not self.config.pred_dataset_path:
            return

        print(f'Load the pred dataset and make predictions that will be saved in the file {self.config.output_pred_path}')
        documents = self.load_dataset(self.config.pred_dataset_path, 'Pred documents')

        diversifier = self.config.pred_diversifier
        batch_size = self.config.training_config.training_arguments.get("per_device_eval_batch_size", 5)

        dataset = self.model.prepare_dataset(documents, diversifier, 'Prepare pred dataset', False, True).prepare_documents()

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=self.model.collate_fn)

        torch.cuda.empty_cache()
        self.model.predict(documents, dataloader, self.config.output_pred_path)

    def save_model(self, *, rewrite: bool = False):
        if self.config.save_path:
            print(f'Save the model in the file {self.config.save_path}')
            self.model.save(path=self.config.save_path, rewrite=rewrite)

    def load_dataset(self, dataset_path: Path, desc: str = "") -> List[Document]:
        return list(tqdm(self.loader.load(dataset_path), desc=desc))

    def set_seed(self):
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
