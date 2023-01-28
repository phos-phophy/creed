from collections import defaultdict
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
        """
        Config's structure:
        {
            "trainer": {
                "log_dir": str
                "learning_rate": float
                "batch_size": int
                "epochs": int
            },

            "save_path": str,

            "model": {
                "load_path": str (optional if other parameters are not specified)
                ...
            }
        }
        """

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
        dev_dataloader = dev_dataset and DataLoader(dev_dataset, batch_size=self.params["batch_size"], shuffle=True, collate_fn=collate_fn)

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
                writer.add_scalar("batch loss / train", loss.item(), epoch * len(train_dataloader) + i)

            avg_loss = np.mean(epoch_loss)
            writer.add_scalar("loss / train", avg_loss, epoch)

            if dev_dataset is None:
                pbar.set_description(f'loss / train: {avg_loss}')
            else:
                dev_score = self.score_model(dataloader=dev_dataloader)
                pbar.set_description(f'macro / f_score / dev: {dev_score.macro_score.f_score}')
                self.save_dev_results(writer, dev_score, epoch)

        self.model.save(path=self.save_path, rewrite=rewrite)

    @staticmethod
    def save_dev_results(writer: SummaryWriter, dev_score: ModelScore, epoch: int):
        def save_results(score_type: str, score: Score):
            writer.add_scalar(f"{score_type} / f_score / dev", score.f_score, epoch)
            writer.add_scalar(f"{score_type} / recall / dev", score.recall, epoch)
            writer.add_scalar(f"{score_type} / precision / dev", score.precision, epoch)

        save_results("macro", dev_score.macro_score)
        save_results("micro", dev_score.micro_score)

        for relation, relation_score in dev_score.relations_score.items():
            save_results(relation, relation_score)

    @torch.no_grad()
    def score_model(self, dataset: AbstractDataset = None, dataloader: DataLoader = None) -> ModelScore:
        if dataloader is None and dataset is None:
            raise ValueError("")

        dataloader = dataloader or DataLoader(dataset, batch_size=self.params["batch_size"], shuffle=True, collate_fn=collate_fn)

        predictions = []
        gold_labels = defaultdict(list)

        self.model.eval()
        for tokens, labels in dataloader:

            if torch.cuda.is_available():
                tokens = {key: token.cuda() for key, token in tokens.items()}

            logits: torch.Tensor = self.model(**tokens).detach().cpu()
            torch.cuda.empty_cache()

            predictions.append(logits)
            for key, item in labels:
                gold_labels[key].append(item)

        return self.model.score(torch.cat(predictions, dim=0), {key: torch.cat(item, dim=0) for key, item in gold_labels.items()})
