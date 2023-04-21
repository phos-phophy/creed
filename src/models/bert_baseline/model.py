import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from src.abstract import AbstractDataset, AbstractModel, DiversifierConfig, Document, NO_REL_IND, cuda_autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertModel

from .datasets import DiversifiedTypedEntityMarkerDataset, EntityMarkerDataset, TypedEntityMarkerDataset


class BertBaseline(AbstractModel):

    def __init__(
            self,
            pretrained_model_path: str,
            tokenizer_path: str,
            inner_model_type: str,
            relations: Iterable[str],
            dropout: float,
            entities: Optional[Iterable[str]] = None
    ):
        super().__init__(relations)
        self._inner_model_type = inner_model_type

        self._entities = tuple(entities) if entities else ('',)

        self._added_vocab_len = 0
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self._encoder: BertModel = AutoModel.from_pretrained(pretrained_model_path)

        self._classifier = torch.nn.Sequential(
            torch.nn.Linear(2 * self._encoder.config.hidden_size, self._encoder.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(self._encoder.config.hidden_size, len(self.relations))
        )

        self._loss_fnt = torch.nn.CrossEntropyLoss()

    @property
    def inner_model_type(self):
        return self._inner_model_type

    @property
    def entities(self):
        return self._entities

    def prepare_dataset(
            self,
            documents: Iterable[Document],
            diversifier: DiversifierConfig,
            desc: str,
            extract_labels: bool = False,
            evaluation: bool = False
    ) -> AbstractDataset:

        if self.inner_model_type == 'entity_marker':
            dataset = EntityMarkerDataset(
                documents, self._tokenizer, extract_labels, evaluation, self.relations, desc, diversifier
            ).prepare_documents()

        elif self.inner_model_type == 'typed_entity_marker':
            dataset = TypedEntityMarkerDataset(
                documents, self._tokenizer, extract_labels, evaluation, self.relations, desc, diversifier
            ).prepare_documents()

        elif self.inner_model_type == 'div_typed_entity_marker':
            dataset = DiversifiedTypedEntityMarkerDataset(
                documents, self._tokenizer, extract_labels, evaluation, self.relations, desc, diversifier
            ).prepare_documents()

        else:
            raise ValueError

        self._encoder.resize_token_embeddings(len(self._tokenizer))

        return dataset

    def evaluate(self, dataloader: DataLoader, output_path: Path = None) -> None:
        self._evaluate(dataloader, output_path, 'Evaluating')

    def predict(self, documents: List[Document], dataloader: DataLoader, output_path: Path) -> None:
        raise NotImplementedError

    def test(self, dataloader: DataLoader, output_path: Path = None) -> None:
        self._evaluate(dataloader, output_path, 'Test')

    @cuda_autocast
    def forward(self, input_ids=None, attention_mask=None, labels=None, ss=None, os=None):
        pooled_output = self._encoder(input_ids, attention_mask=attention_mask)[0]  # (bs, length, hidden_size)

        idx = torch.arange(input_ids.size(0)).to(input_ids.device)  # (bs, )
        ss_emb = pooled_output[idx, ss.flatten()]  # (bs, hidden_size)
        os_emb = pooled_output[idx, os.flatten()]  # (bs, hidden_size)

        h = torch.cat((ss_emb, os_emb), dim=-1)  # (bs, 2 * hidden_size)
        logits = self._classifier(h)  # (bs, num_rel)

        outputs = (logits,)
        if labels is not None:
            loss = self._loss_fnt(logits.float(), labels.flatten())
            outputs = (loss,) + outputs

        return outputs

    def _evaluate(self, dataloader: DataLoader, output_path: Path = None, desc: str = None) -> None:

        if Path is None:
            pass

        loss = 0.0
        labels_ids, preds = [], []
        for inputs in tqdm(dataloader, desc=desc):
            self.eval()

            inputs = {key: token.cuda() for key, token in inputs.items()} if torch.cuda.is_available() else inputs
            labels_ids += inputs["labels"].flatten().tolist() if "labels" in inputs else [None]

            with torch.no_grad():
                outputs = self(**inputs)
                if len(outputs) > 1:
                    batch_loss, logits = outputs
                    loss += batch_loss.mean().item()
                else:
                    logits = outputs

                pred = torch.argmax(logits, dim=-1)
            preds += pred.tolist()

        labels_ids = np.array(labels_ids, dtype=np.int64)  # (N,)
        preds = np.array(preds, dtype=np.int64)  # (N,)

        self._count_stats(labels_ids, preds, loss, output_path)

    def _count_stats(self, labels_ids: np.ndarray, preds: np.ndarray, loss: float, output_path: Path):

        labels = np.arange(len(self.relations))
        relations = list(self.relations)

        # without NO_REL relations
        labels = labels[labels != NO_REL_IND]
        relations = relations[:NO_REL_IND] + relations[NO_REL_IND + 1:]

        pr, r, f, _ = precision_recall_fscore_support(labels_ids, preds, average='micro', labels=labels, zero_division=0)
        pr_sep, r_sep, f_sep, _ = precision_recall_fscore_support(labels_ids, preds, average=None, labels=labels, zero_division=0)

        result = {
            "loss": float(loss),
            "precision": float(pr),
            "recall": float(r),
            "f1": float(f),
            "labels": {
                relation: {
                    "precision": pr_sep[ind],
                    "recall": r_sep[ind],
                    "f1": f_sep[ind]
                } for ind, relation in enumerate(relations)}
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w') as file:
            json.dump(result, file, indent=4)
