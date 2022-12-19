from pathlib import Path

import torch
from src.abstract import AbstractDataset, AbstractModel, collate_fn
from src.models import get_model
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(self, config):
        self.save_path = Path(config["save_path"])
        self.params = config["trainer"]

        if "load_path" in config["model"]:
            self.model: AbstractModel = AbstractModel.load(config["model"]["load_path"])
        else:
            self.model: AbstractModel = get_model(**config["model"])

    def train_model(self, train_dataset: AbstractDataset, dev_dataset: AbstractDataset):

        val_accuracy = []

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params["learning_rate"])

        train_dataloader = DataLoader(train_dataset, batch_size=self.params["batch_size"], shuffle=True, collate_fn=collate_fn)

        pbar = tqdm(range(self.params["epochs"]), total=self.params["epochs"])

        for _ in pbar:
            self.model.train()

            for tokens, labels in train_dataloader:

                if torch.cuda.is_available():
                    tokens = {key: token.cuda() for key, token in tokens.items()}
                    labels = {key: label.cuda() for key, label in labels.items()}

                self.model.zero_grad()
                logits = self.model(**tokens)
                loss = self.model.compute_loss(logits=logits, **labels)
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

            with torch.no_grad():
                acc = self.score_model(dev_dataset)
                val_accuracy.append(acc)
                self.model.train()

            pbar.set_description(f'Val accuracy {acc:.5f}')

        self.model.save(path=self.save_path, rewrite=False)

    def score_model(self, dataset: AbstractDataset):
        self.model.eval()

        predictions = []
        gold_labels = []

        for tokens, labels in dataset:

            tokens = tokens.unsqueeze(0)

            if torch.cuda.is_available():
                tokens = {key: token.cuda() for key, token in tokens.items()}
                labels = {key: label.cuda() for key, label in labels.items()}

            logits = self.model(**tokens)

            predictions.append(logits)
            gold_labels.append(labels)

        score = self.model.score(predictions, gold_labels)

        return score
