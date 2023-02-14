import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from src.abstract import AbstractDataset, AbstractWrapperModel, Document
from torch.utils.data import DataLoader
from tqdm import tqdm

from .inner_models import AbstractSSANAdaptInnerModel, get_inner_model


class SSANAdaptModel(AbstractWrapperModel):

    def __init__(
            self,
            entities: Iterable[str],
            relations: Iterable[str],
            inner_model_type: str,
            hidden_dim: int,
            dropout: float,
            **kwargs
    ):
        self._entities = tuple(entities)

        super(SSANAdaptModel, self).__init__(relations)

        self._inner_model: AbstractSSANAdaptInnerModel = get_inner_model(
            inner_model_type=inner_model_type, entities=entities, relations=relations, **kwargs
        )

        self._dist_ceil = self._inner_model.dist_ceil
        self._dist_emb_dim = self._dist_ceil * 2

        out_dim = next(module.out_features for module in list(self._inner_model.modules())[::-1] if "out_features" in module.__dict__)

        self._dim_reduction = torch.nn.Linear(out_dim, hidden_dim)
        self._dropout = torch.nn.Dropout(dropout)
        self._rel_dist_embeddings = torch.nn.Embedding(self._dist_emb_dim, self._dist_emb_dim, padding_idx=self._dist_ceil)
        self._bili = torch.nn.Bilinear(hidden_dim + self._dist_emb_dim, hidden_dim + self._dist_emb_dim, len(self.relations))

        self.threshold = None

    @property
    def entities(self):
        return self._entities

    def forward(
            self,
            dist_ids=None,  # (bs, max_ent, max_ent)
            ent_mask=None,  # (bs, max_ent, len)
            labels=None,  # (bs, max_ent, max_ent, num_link)
            labels_mask=None,  # (bs, max_ent, max_ent)
            **kwargs
    ) -> Any:

        output = self._inner_model(**kwargs)

        # tensors for each token in the text
        tokens: torch.Tensor = output[0]  # (bs, len, dim)
        tokens: torch.Tensor = torch.relu(self._dim_reduction(tokens))  # (bs, len, r_dim)

        # extract tensors for entities only
        entity: torch.Tensor = torch.matmul(ent_mask.float(), tokens)  # (bs, max_ent, r_dim)

        # define head and tail entities (head --|relation|--> tail)
        max_ent = ent_mask.size()[1]
        h_entity: torch.Tensor = entity[:, :, None, :].repeat(1, 1, max_ent, 1)  # (bs, max_ent, max_ent, r_dim)
        t_entity: torch.Tensor = entity[:, None, :, :].repeat(1, max_ent, 1, 1)  # (bs, max_ent, max_ent, r_dim)

        # add information about the relative distance between entities
        dist_ids += self._dist_ceil
        h_entity = torch.cat([h_entity, self._rel_dist_embeddings(dist_ids)], dim=-1)
        t_entity = torch.cat([t_entity, self._rel_dist_embeddings((self._dist_emb_dim - dist_ids) % self._dist_emb_dim)], dim=-1)

        h_entity: torch.Tensor = self._dropout(h_entity)
        t_entity: torch.Tensor = self._dropout(t_entity)

        # get prediction  (without function activation)
        logits: torch.Tensor = self._bili(h_entity, t_entity)  # (bs, max_ent, max_ent, num_links)

        if labels is not None:
            return self._compute_loss(logits, labels, labels_mask), torch.sigmoid(logits)

        return torch.sigmoid(logits)

    def prepare_dataset(self, document: Iterable[Document], extract_labels=False, evaluation=False) -> AbstractDataset:
        return self._inner_model.prepare_dataset(document, extract_labels, evaluation)

    def _compute_loss(
            self,
            logits: torch.Tensor,  # (bs, max_ent, max_ent, num_link)
            labels: torch.Tensor,  # (bs, max_ent, max_ent, num_link)
            labels_mask: torch.Tensor,  # (bs, max_ent, max_ent)
    ):
        max_ent = logits.shape[1]
        num_links = len(self.relations)

        pair_logits = logits.view(-1, num_links)  # (bs * max_ent ^ 2, num_link)
        pair_labels = labels.float().view(-1, num_links)  # (bs * max_ent ^ 2, num_link)

        loss_function = torch.nn.BCEWithLogitsLoss(reduction="none")

        pair_loss = loss_function(pair_logits, pair_labels)  # (bs * max_ent ^ 2, num_link)
        mean_pair_loss = pair_loss.view(-1, max_ent, max_ent, num_links)  # (bs, max_ent, max_ent, num_link)
        mean_pair_loss = torch.mean(mean_pair_loss, dim=-1)  # (bs, max_ent, max_ent)

        batch_loss = torch.sum(mean_pair_loss * labels_mask, dim=[1, 2]) / torch.sum(labels_mask, dim=[1, 2])  # (bs,)
        mean_batch_loss = torch.mean(batch_loss)

        return mean_batch_loss

    def evaluate(self, dataloader: DataLoader, output_path: str = None):

        loss = 0.0
        preds, ent_masks, labels_ids = [], [], []

        for _, inputs in tqdm(dataloader, desc='Evaluating'):
            self.model.eval()
            inputs = {key: token.cuda() for key, token in inputs.items()} if torch.cuda.is_available() else inputs

            with torch.no_grad():
                batch_loss, logits = self.model(**inputs)
                loss += batch_loss.mean().item()

            preds.append(logits.detach().cpu().numpy())
            ent_masks.append(inputs["ent_mask"].detach().cpu().numpy())
            labels_ids.append(inputs["labels"].detach().cpu().numpy())

        loss /= len(dataloader)
        preds, ent_masks, labels_ids = np.vstack(preds), np.vstack(ent_masks), np.vstack(labels_ids)

        total_labels = torch.sum(labels_ids[:, :, :, 1:]).item()
        output_preds = []

        # pred: (ent, ent, num_link)
        # ent_mask: (ent, len)
        # labels: (ent, ent, num_link)
        for (pred, ent_mask, labels) in zip(preds, ent_masks, labels_ids):

            # don't take into account <NO_REL> relation
            gold_relations = [(h, t, r + 1) for (h, t, r) in zip(*np.where(labels[:, :, 1:]))]

            for h in range(pred.shape[0]):
                for t in range(pred.shape[0]):

                    if h == t or np.all(ent_mask[h] == 0) or np.all(ent_mask[t] == 0):
                        continue

                    for predicate_id, logit in enumerate(pred[h][t]):
                        if predicate_id == 0:
                            continue

                        is_right_relation = (h, t, predicate_id) in gold_relations
                        output_preds.append((is_right_relation, logit, h, t, predicate_id))

        output_preds.sort(key=lambda x: x[1], reverse=True)

        pr_x = []
        pr_y = []
        correct = 0
        for i, pred in enumerate(output_preds, start=1):
            correct += pred[0]
            pr_y.append(float(correct) / i)
            pr_x.append(float(correct) / total_labels)

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()
        f1_pos = f1_arr.argmax()
        threshold = output_preds[f1_pos][1]

        result = {
            "loss": loss,
            "precision": pr_y[f1_pos],
            "recall": pr_x[f1_pos],
            "f1": f1,
            "threshold": threshold
        }

        with open(Path(output_path), 'r') as file:
            json.dump(result, file)

        self.threshold = threshold

    def predict(self, dataloader: DataLoader, output_path: str = None):
        if self.threshold is None:
            raise ValueError("First calculate the threshold value using evaluate function (on dev dataset)!")
