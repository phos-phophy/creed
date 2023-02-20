import json
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import torch
from src.abstract import AbstractDataset, AbstractWrapperModel, Document, NO_REL_IND
from torch.utils.data import DataLoader
from tqdm import tqdm

from .inner_models import AbstractSSANAdaptInnerModel, get_inner_model


class SSANAdaptModel(AbstractWrapperModel):

    def __init__(
            self,
            inner_model_type: str,
            hidden_dim: int,
            relations: Iterable[str],
            **kwargs
    ):

        super(SSANAdaptModel, self).__init__(relations)

        self._inner_model: AbstractSSANAdaptInnerModel = get_inner_model(inner_model_type=inner_model_type, relations=relations, **kwargs)

        self._dist_ceil = self._inner_model.dist_ceil
        self._dist_emb_dim = self._dist_ceil * 2

        out_dim = next(module.out_features for module in list(self._inner_model.modules())[::-1] if "out_features" in module.__dict__)

        self._dim_reduction = torch.nn.Linear(out_dim, hidden_dim)
        self._dropout = torch.nn.Dropout(self._inner_model.config.hidden_dropout_prob)
        self._rel_dist_embeddings = torch.nn.Embedding(self._dist_emb_dim, self._dist_emb_dim, padding_idx=self._dist_ceil)
        self._bili = torch.nn.Bilinear(hidden_dim + self._dist_emb_dim, hidden_dim + self._dist_emb_dim, len(self.relations))

        self._threshold = None

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

    def _get_preds(self, dataloader: DataLoader, desc: str):

        loss = 0.0
        preds, ent_masks, labels_ids = [], [], []

        for inputs in tqdm(dataloader, desc=desc):
            self.eval()
            inputs = {key: token.cuda() for key, token in inputs.items()} if torch.cuda.is_available() else inputs

            with torch.no_grad():
                outputs = self(**inputs)
                if isinstance(outputs, tuple):
                    batch_loss, logits = outputs
                    loss += batch_loss.mean().item()
                else:
                    logits = outputs

            preds.append(logits.detach().cpu().numpy())
            ent_masks.append(inputs["ent_mask"].detach().cpu().numpy())
            labels_ids.append(inputs["labels"].detach().cpu().numpy() if "labels" in inputs else None)

        def zero_array(array: np.ndarray, length: int):
            return np.zeros((array.shape[0], array.shape[1], length - array.shape[2]))

        max_len = max(map(lambda ent_mask: ent_mask.shape[2], ent_masks))
        ent_masks = [np.concatenate((ent_mask, zero_array(ent_mask, max_len)), axis=-1) for ent_mask in ent_masks]

        loss /= len(dataloader)
        preds = np.vstack(preds)  # (N, ent, ent, num_link)
        ent_masks = np.vstack(ent_masks)  # (N, ent, len)
        labels_ids = np.vstack(labels_ids)  # (N, ent, ent, num_link)

        return loss, preds, ent_masks, labels_ids

    def evaluate(self, dataloader: DataLoader, output_path: Path = None):

        # don't take into account gold <NO_REL> relation
        loss, preds, ent_masks, labels_ids = self._get_preds(dataloader, 'Evaluating')
        labels_ids = torch.cat((labels_ids[:, :, :, :NO_REL_IND], labels_ids[:, :, :, NO_REL_IND + 1:]), dim=-1)

        total_labels = np.sum(labels_ids)
        output_preds = []

        for pred, ent_mask, gold_labels in zip(preds, ent_masks, labels_ids):

            # don't take into account gold <NO_REL> relation
            gold_relations = [(h, t, r + 1) for h, t, r in zip(*np.where(gold_labels))]

            for h, t, logit, predicate_id in iter_over_pred(pred, ent_mask, 0):
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
            "loss": float(loss),
            "precision": float(pr_y[f1_pos]),
            "recall": float(pr_x[f1_pos]),
            "f1": float(f1),
            "threshold": float(threshold)
        }

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open('w') as file:
                json.dump(result, file)

        self._threshold = threshold

    def predict(self, documents: List[Document], dataloader: DataLoader, output_path: Path):
        if self._threshold is None:
            raise ValueError("First calculate the threshold value using evaluate function (on dev dataset)!")

        _, preds, ent_masks, _ = self._get_preds(dataloader, 'Predicting')

        def build_docred_pred(title, h_idx, t_idx, r):
            return {"title": title, "h_idx": h_idx, "t_idx": t_idx, "r": r, "evidence": []}

        output_preds = []
        for document, pred, ent_mask in zip(documents, preds, ent_masks):
            for h, t, predicate_id in iter_over_pred(pred, ent_mask, self._threshold):
                output_preds.append(build_docred_pred(document.doc_id, h, t, self.relations[predicate_id]))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w') as file:
            json.dump(output_preds, file)


def iter_over_pred(pred, ent_mask, threshold):
    for h in range(pred.shape[0]):
        for t in range(pred.shape[0]):

            if h == t or np.all(ent_mask[h] == 0) or np.all(ent_mask[t] == 0):
                continue

            for predicate_id, logit in enumerate(pred[h][t]):
                if predicate_id == 0:
                    continue

                if logit >= threshold:
                    yield h, t, predicate_id
