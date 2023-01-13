from pathlib import Path

import numpy as np
import torch
from src.abstract import AbstractDataset, AbstractModel, ModelScore, Score, collate_fn
from src.models import get_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    def __init__(self, config):
        self.save_path = Path(config["save_path"])
        self.params = config["trainer"]

        if "load_path" in config["model"]:
            self.model: AbstractModel = AbstractModel.load(config["model"]["load_path"])
        else:
            self.model: AbstractModel = get_model(**config["model"])

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def train_model(self, train_dataset: AbstractDataset, dev_dataset: AbstractDataset = None, rewrite: bool = False):
        writer = SummaryWriter(log_dir=self.params["log_dir"])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params["learning_rate"])

        train_dataloader = DataLoader(train_dataset, batch_size=self.params["batch_size"], shuffle=True, collate_fn=collate_fn)

        pbar = tqdm(range(self.params["epochs"]), total=self.params["epochs"])

        for epoch in pbar:
            epoch_loss = []
            self.model.train()

            for i, (tokens, labels) in enumerate(train_dataloader):

                if torch.cuda.is_available():
                    tokens = {key: token.cuda() for key, token in tokens.items()}
                    labels = {key: label.cuda() for key, label in labels.items()}

                self.model.zero_grad()
                logits = self.model(**tokens)
                loss = self.model.compute_loss(logits=logits, **labels)
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                epoch_loss.append(loss.item())
                writer.add_scalar(
                    "batch loss / train", loss.item(), epoch * len(train_dataloader) + i
                )

            avg_loss = np.mean(epoch_loss)
            writer.add_scalar("loss / train", avg_loss, epoch)

            if dev_dataset is None:
                pbar.set_description(f'loss / train: {avg_loss}')
            else:
                with torch.no_grad():
                    dev_score = self.score_model(dev_dataset)
                    self.model.train()

                pbar.set_description(f'macro / f_score / dev: {dev_score.macro_score.f_score}')

                self.save_results(writer, dev_score.macro_score, "macro", epoch)
                self.save_results(writer, dev_score.micro_score, "micro", epoch)

                for relation, relation_score in dev_score.relations_score.items():
                    self.save_results(writer, relation_score, relation, epoch)

        self.model.save(path=self.save_path, rewrite=rewrite)

    @staticmethod
    def save_results(writer: SummaryWriter, score: Score, score_type: str, epoch: int):
        writer.add_scalar(f"{score_type} / f_score / dev", score.f_score, epoch)
        writer.add_scalar(f"{score_type} / recall / dev", score.recall, epoch)
        writer.add_scalar(f"{score_type} / precision / dev", score.precision, epoch)

    def score_model(self, dataset: AbstractDataset) -> ModelScore:
        self.model.eval()

        predictions = []
        gold_labels = []

        for tokens, labels in dataset:

            tokens = {key: token.unsqueeze(0) for key, token in tokens.items()}

            if torch.cuda.is_available():
                tokens = {key: token.cuda() for key, token in tokens.items()}

            logits: torch.Tensor = self.model(**tokens).detach().cpu()
            torch.cuda.empty_cache()

            predictions.append(logits)
            gold_labels.append(labels)

        return self.model.score(predictions, gold_labels)
