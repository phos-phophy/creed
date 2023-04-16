import json
from functools import wraps
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from src.abstract import AbstractDataset, AbstractWrapperModel, DiversifierConfig, Document, NO_ENT_IND, NO_REL_IND
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings


def new_forward(embeddings: BertEmbeddings, old_forward):
    @wraps(old_forward)
    def forward(**kwargs):
        return old_forward(**kwargs) + embeddings.ner_embeddings(embeddings.ner_ids)
    return forward


class BertBaseline(AbstractWrapperModel):

    def __init__(self, pretrained_model_path: str, relations: Iterable[str], entities: Iterable[str], dropout: float):
        super().__init__(relations)

        self._entities = tuple(entities)

        self._encoder: BertModel = AutoModel.from_pretrained(pretrained_model_path)
        self._modify_embeddings()

        self._classifier = torch.nn.Sequential(
            torch.nn.Linear(2 * self._encoder.config.hidden_size, self._encoder.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(self._encoder.config.hidden_size, len(self.relations))
        )

        self._loss_fnt = torch.nn.CrossEntropyLoss()

    def _modify_embeddings(self):
        embedding_dim = self._encoder.embeddings.word_embeddings.embedding_dim
        ner_embeddings = torch.nn.Embedding(num_embeddings=len(self._entities), embedding_dim=embedding_dim, padding_idx=NO_ENT_IND)

        self._encoder.embeddings.ner_embeddings = ner_embeddings
        self._encoder.embeddings.forward = new_forward(self._encoder.embeddings, self._encoder.embeddings.forward)

    def prepare_dataset(
            self,
            documents: Iterable[Document],
            diversifier: DiversifierConfig,
            desc: str,
            extract_labels: bool = False,
            evaluation: bool = False,
            cache_dir: Path = None,
            dataset_name: str = ''
    ) -> AbstractDataset:
        pass

    def evaluate(self, dataloader: DataLoader, output_path: Path = None) -> None:
        self._evaluate(dataloader, output_path, 'Evaluating')

    def predict(self, documents: List[Document], dataloader: DataLoader, output_path: Path) -> None:
        raise NotImplementedError

    def test(self, dataloader: DataLoader, output_path: Path = None) -> None:
        self._evaluate(dataloader, output_path, 'Test')

    def forward(self, input_ids=None, ner_ids=None, attention_mask=None, labels=None, ss=None, os=None):
        self._encoder.embeddings.ner_ids = ner_ids
        pooled_output = self._encoder(input_ids, attention_mask=attention_mask)[0]  # (bs, length, hidden_size)

        idx = torch.arange(input_ids.size(0)).to(input_ids.device)  # (bs, )
        ss_emb = pooled_output[idx, ss]  # (bs, hidden_size)
        os_emb = pooled_output[idx, os]  # (bs, hidden_size)

        h = torch.cat((ss_emb, os_emb), dim=-1)  # (bs, 2 * hidden_size)
        logits = self._classifier(h)  # (bs, num_rel)

        outputs = (logits,)
        if labels is not None:
            loss = self._loss_fnt(logits.float(), labels)
            outputs = (loss,) + outputs

        return outputs

    def _evaluate(self, dataloader: DataLoader, output_path: Path = None, desc: str = None) -> None:
        loss = 0.0
        labels_ids, preds = [], []
        for inputs in tqdm(dataloader, desc=desc):
            self.eval()

            inputs = {key: token.cuda() for key, token in inputs.items()} if torch.cuda.is_available() else inputs
            labels_ids += inputs["labels"].tolist() if "labels" in inputs else [None]

            with torch.no_grad():
                outputs = self(**inputs)
                if len(outputs) > 1:
                    batch_loss, logits = outputs
                    loss += batch_loss.mean().item()
                else:
                    logits = outputs

                pred = torch.argmax(logits, dim=-1)
            preds += pred.tolist()

        labels_ids = np.array(labels_ids, dtype=np.int64)
        preds = np.array(preds, dtype=np.int64)

        correct_by_relation = ((labels_ids == preds) & (preds != NO_REL_IND)).astype(np.int32).sum()
        guessed_by_relation = (preds != 0).astype(np.int32).sum()
        gold_by_relation = (labels_ids != 0).astype(np.int32).sum()

        prec_micro = 1.0
        if guessed_by_relation > 0:
            prec_micro = float(correct_by_relation) / float(guessed_by_relation)

        recall_micro = 1.0
        if gold_by_relation > 0:
            recall_micro = float(correct_by_relation) / float(gold_by_relation)

        f1_micro = 0.0
        if prec_micro + recall_micro > 0.0:
            f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)

        result = {
            "loss": float(loss),
            "precision": float(prec_micro),
            "recall": float(recall_micro),
            "f1": float(f1_micro)
        }

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open('w') as file:
                json.dump(result, file)
