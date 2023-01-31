from typing import Dict, Sequence

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import EvalPrediction


def score_model(eval_prediction: EvalPrediction, relations: Sequence[str]) -> Dict[str, float]:

    labels: torch.Tensor = torch.from_numpy(eval_prediction.label_ids[0])  # (N, max_ent, max_ent, num_link)
    labels_mask: torch.Tensor = torch.from_numpy(eval_prediction.label_ids[1])  # (N, max_ent, max_ent)
    logits: torch.Tensor = torch.from_numpy(eval_prediction.predictions)  # (N, max_ent, max_ent, num_link)

    if logits.shape != labels.shape:
        raise ValueError(f"Logits and gold labels have incompatible shapes: {logits.shape} and {labels.shape} respectively")

    num_links = logits.shape[-1]

    if len(relations) != num_links:
        raise ValueError(f"Number of relations and predicted links are different: {len(relations)} and {num_links} respectively")

    labels_mask = labels_mask.view(-1, 1)  # (N * max_ent ^ 2, 1)
    pair_logits = logits.view(-1, num_links)  # (N * max_ent ^ 2, num_link)
    pair_labels = labels.float().view(-1, num_links)  # (N * max_ent ^ 2, num_link)

    # remove fake pairs
    pair_logits = pair_logits * labels_mask
    pair_labels = pair_labels * labels_mask

    pair_logits_ind = torch.argmax(pair_logits, dim=-1)  # (N * max_ent ^ 2)
    pair_labels_ind = torch.argmax(pair_labels, dim=-1)  # (N * max_ent ^ 2)

    score = dict()

    pr, r, f, _ = precision_recall_fscore_support(
        pair_labels_ind, pair_logits_ind, average=None, labels=list(range(num_links)), zero_division=0
    )

    for ind, relation_name in enumerate(relations):
        score[f"{relation_name} / precision"] = pr[ind]
        score[f"{relation_name} / recall"] = r[ind]
        score[f"{relation_name} / f_score"] = f[ind]

    score["macro / precision"] = np.mean(pr).item()
    score["macro / recall"] = np.mean(r).item()
    score["macro / f_score"] = np.mean(f).item()

    pr, r, f, _ = precision_recall_fscore_support(
        pair_labels_ind, pair_logits_ind, average='micro', labels=list(range(num_links)), zero_division=0
    )

    score["micro / precision"] = pr
    score["micro / recall"] = r
    score["micro / f_score"] = f

    return score
