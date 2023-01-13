from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from src.abstract import AbstractDataset, AbstractModel, Document, ModelScore, Score

from .inner_models import AbstractSSANAdaptInnerModel, get_inner_model


class SSANAdaptModel(AbstractModel):

    def __init__(
            self,
            entities: Iterable[str],
            relations: Iterable[str],
            inner_model_type: str,
            hidden_dim: int,
            dropout: float,
            no_ent_ind: int = None,
            **kwargs
    ):
        if no_ent_ind is None:
            no_ent_ind = 0
            entities = ['<NO_ENT>'] + list(entities)

        self._no_ent_ind = no_ent_ind
        self._entities = tuple(entities)

        super(SSANAdaptModel, self).__init__(relations)

        self._inner_model: AbstractSSANAdaptInnerModel = get_inner_model(inner_model_type=inner_model_type, entities=entities,
                                                                         no_ent_ind=no_ent_ind, relations=relations, **kwargs)

        self._dist_ceil = self._inner_model.dist_ceil
        self._dist_emb_dim = self._dist_ceil * 2

        out_dim = next(module.out_features for module in list(self._inner_model.modules())[::-1] if "out_features" in module.__dict__)

        self._dim_reduction = torch.nn.Linear(out_dim, hidden_dim)
        self._dropout = torch.nn.Dropout(dropout)
        self._rel_dist_embeddings = torch.nn.Embedding(self._dist_emb_dim, self._dist_emb_dim, padding_idx=self._dist_ceil)
        self._bili = torch.nn.Bilinear(hidden_dim + self._dist_emb_dim, hidden_dim + self._dist_emb_dim, len(self.relations))

        self._loss = torch.nn.BCEWithLogitsLoss(reduction="none")

    @property
    def entities(self):
        return self._entities

    @property
    def no_ent_ind(self):
        return self._no_ent_ind

    def forward(
            self,
            dist_ids=None,  # (bs, max_ent, max_ent)
            ent_mask=None,  # (bs, max_ent, len)
            **kwargs
    ) -> Any:

        output = self._inner_model(**kwargs)

        # tensors for each token in the text
        tokens: torch.Tensor = output[0]  # (bs, len, dim)
        tokens: torch.Tensor = torch.relu(self._dim_reduction(tokens))  # (bs, len, r_dim)

        # extract tensors for entities only
        entity: torch.Tensor = torch.matmul(ent_mask.float(), tokens)  # (bs, max_ent, r_dim)

        # define head and tail entities (head --|relation|--> tail)
        h_entity: torch.Tensor = entity[:, :, None, :].repeat(1, 1, ent_mask.size()[1], 1)  # (bs, max_ent, max_ent, r_dim)
        t_entity: torch.Tensor = entity[:, None, :, :].repeat(1, ent_mask.size()[1], 1, 1)  # (bs, max_ent, max_ent, r_dim)

        # add information about the relative distance between entities
        dist_ids += self._dist_ceil
        h_entity = torch.cat([h_entity, self._rel_dist_embeddings(dist_ids)], dim=-1)
        t_entity = torch.cat([t_entity, self._rel_dist_embeddings((self._dist_emb_dim - dist_ids) % self._dist_emb_dim)], dim=-1)

        h_entity: torch.Tensor = self._dropout(h_entity)
        t_entity: torch.Tensor = self._dropout(t_entity)

        # get prediction  (without function activation)
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
    ) -> ModelScore:
        num_links = len(self.relations)

        labels: List[torch.Tensor] = list(map(lambda x: x["labels"], gold_labels))  # (max_ent, max_ent, num_link)
        labels_mask: List[torch.Tensor] = list(map(lambda x: x["labels_mask"], gold_labels))  # (max_ent, max_ent)

        logits: torch.Tensor = torch.cat(logits, dim=0)  # (bs, max_ent, max_ent, num_link)
        labels: torch.Tensor = torch.stack(labels, dim=0)  # (bs, max_ent, max_ent, num_link)
        labels_mask: torch.Tensor = torch.stack(labels_mask, dim=0)  # (bs, max_ent, max_ent)

        pair_logits = logits.view(-1, num_links)  # (bs * max_ent ^ 2, num_link)
        pair_labels = labels.float().view(-1, num_links)  # (bs * max_ent ^ 2, num_link)
        labels_mask = labels_mask.view(-1, 1)  # (bs * max_ent ^ 2, 1)

        # remove fake pairs
        pair_logits = pair_logits * labels_mask
        pair_labels = pair_labels * labels_mask

        pair_logits_ind = torch.argmax(pair_logits, dim=-1)  # (bs * max_ent ^ 2)
        pair_labels_ind = torch.argmax(pair_labels, dim=-1)  # (bs * max_ent ^ 2)

        labels = list(range(num_links))
        precision, recall, f_score, _ = precision_recall_fscore_support(pair_labels_ind, pair_logits_ind, average=None, labels=labels,
                                                                        zero_division=0)

        relations_score = dict()
        for ind, relation_name in enumerate(self.relations):
            relations_score[relation_name] = Score(precision=precision[ind], recall=recall[ind], f_score=f_score[ind])

        macro_score = Score(precision=np.mean(precision).item(), recall=np.mean(recall).item(), f_score=np.mean(f_score).item())

        precision, recall, f_score, _ = precision_recall_fscore_support(pair_labels_ind, pair_logits_ind, average='micro', labels=labels)
        micro_score = Score(precision=precision, recall=recall, f_score=f_score)

        return ModelScore(relations_score=relations_score, macro_score=macro_score, micro_score=micro_score)
