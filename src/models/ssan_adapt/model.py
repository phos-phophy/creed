from collections import defaultdict
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from src.abstract import AbstractDataset, AbstractModel, Document

from .inner_models import get_inner_model


class SSANAdaptModel(AbstractModel):

    def __init__(
            self,
            entities: Iterable[str],
            relations: Iterable[str],
            model_type: str,
            hidden_dim: int,
            dropout: float,
            **kwargs
    ):
        super(SSANAdaptModel, self).__init__(entities, relations)

        self._inner_model: AbstractModel = get_inner_model(model_type=model_type, entities=entities, relations=relations, **kwargs)

        out_dim = next(module.out_features for module in list(self._inner_model.modules())[::-1] if "out_features" in module.__dict__)

        self._dim_reduction = torch.nn.Linear(out_dim, hidden_dim)
        self._dropout = torch.nn.Dropout(dropout)
        self._rel_dist_embeddings = torch.nn.Embedding(20, 20, padding_idx=10)
        self._bili = torch.nn.Bilinear(hidden_dim + 20, hidden_dim + 20, len(self.relations))

        self._loss = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(
            self,
            input_ids=None,  # (bs, len)
            ner_ids=None,  # (bs, len)
            dist_ids=None,  # (bs, max_ent, max_ent)
            ent_mask=None,  # (bs, max_ent, len)
            attention_mask=None,  # (bs, len)
            struct_matrix=None,  # (bs, 5, len, len)
    ) -> Any:

        output = self._inner_model(input_ids=input_ids, ner_ids=ner_ids, attention_mask=attention_mask, struct_matrix=struct_matrix)

        # tensors for each token in the text
        tokens: torch.Tensor = output[0]  # (bs, len, dim)
        tokens: torch.Tensor = torch.relu(self._dim_reduction(tokens))  # (bs, len, r_dim)

        # extract tensors for entities only
        entity: torch.Tensor = torch.matmul(ent_mask.float(), tokens)  # (bs, max_ent, r_dim)

        # define head and tail entities (head --|relation|--> tail)
        h_entity: torch.Tensor = entity[:, :, None, :].repeat(1, 1, ent_mask.size()[1], 1)  # (bs, max_ent, max_ent, r_dim)
        t_entity: torch.Tensor = entity[:, None, :, :].repeat(1, ent_mask.size()[1], 1, 1)  # (bs, max_ent, max_ent, r_dim)

        # add information about the relative distance between entities
        h_entity = torch.cat([h_entity, self._rel_dist_embeddings(dist_ids)], dim=-1)
        t_entity = torch.cat([t_entity, self._rel_dist_embeddings((20 - dist_ids) % 20)], dim=-1)

        h_entity: torch.Tensor = self._dropout(h_entity)
        t_entity: torch.Tensor = self._dropout(t_entity)

        # get prediction
        logits: torch.Tensor = self._bili(h_entity, t_entity)  # (bs, max_ent, max_ent, num_links)

        return logits

    def prepare_dataset(self, document: Iterable[Document], extract_labels=False, evaluation=False) -> AbstractDataset:
        return self._inner_model.prepare_dataset(document, extract_labels, evaluation)

    def compute_loss(
            self,
            logits: torch.Tensor,  # (bs, max_ent, max_ent, num_link)
            labels: torch.Tensor,  # (bs, max_ent, max_ent, num_link)
            labels_mask: torch.Tensor,  # (bs, max_ent, max_ent)
    ):
        max_ent = logits.shape[1]
        num_links = len(self.relations)

        pair_logits = logits.view(-1, num_links)  # (bs * max_ent ^ 2, num_link)
        pair_labels = labels.float().view(-1, num_links)  # (bs * max_ent ^ 2, num_link)

        pair_loss = self._loss(pair_logits, pair_labels)  # (bs * max_ent ^ 2, num_link)
        mean_pair_loss = pair_loss.view(-1, max_ent, max_ent, num_links)  # (bs, max_ent, max_ent, num_link)
        mean_pair_loss = torch.mean(mean_pair_loss, dim=-1)  # (bs, max_ent, max_ent)

        batch_loss = torch.sum(mean_pair_loss * labels_mask, dim=[1, 2]) / torch.sum(labels_mask, dim=[1, 2])  # (bs,)
        mean_batch_loss = torch.mean(batch_loss)

        return mean_batch_loss

    def score(
            self,
            logits: List[torch.Tensor],  # (1, max_ent, max_ent, num_link)
            gold_labels: List[Dict[str, torch.Tensor]]
    ) -> Any:
        num_links = len(self.relations)

        labels: List[torch.Tensor] = list(map(lambda x: x["labels"], gold_labels))  # (max_ent, max_ent, num_link)
        labels_mask: List[torch.Tensor] = list(map(lambda x: x["labels_mask"], gold_labels))  # (max_ent, max_ent)

        logits: torch.Tensor = torch.cat(logits, dim=0)  # (bs, max_ent, max_ent, num_link)
        labels: torch.Tensor = torch.stack(labels, dim=0)  # (bs, max_ent, max_ent, num_link)
        labels_mask: torch.Tensor = torch.stack(labels_mask, dim=0)  # (bs, max_ent, max_ent)

        pair_logits = logits.view(-1, num_links)  # (bs * max_ent ^ 2, num_link)
        pair_labels = labels.view(-1, num_links)  # (bs * max_ent ^ 2, num_link)
        labels_mask = labels_mask.view(-1, 1)  # (bs * max_ent ^ 2, 1)

        # remove fake pairs
        pair_logits = pair_logits * labels_mask
        pair_labels = pair_labels * labels_mask

        pair_logits_ind = torch.argmax(pair_logits, dim=-1)  # (bs * max_ent ^ 2)
        pair_labels_ind = torch.argmax(pair_labels, dim=-1)  # (bs * max_ent ^ 2)

        labels = list(range(num_links))
        precision, recall, f_score, _ = precision_recall_fscore_support(pair_labels_ind, pair_logits_ind, average=None, labels=labels)

        metrics = defaultdict(dict)
        for ind, relation_name in enumerate(self.relations):
            metrics[relation_name]["precision"] = precision[ind]
            metrics[relation_name]["recall"] = recall[ind]
            metrics[relation_name]["f_score"] = f_score[ind]

        metrics["general_macro_scores"]["precision"] = np.mean(precision)
        metrics["general_macro_scores"]["recall"] = np.mean(recall)
        metrics["general_macro_scores"]["f_score"] = np.mean(f_score)

        return metrics
