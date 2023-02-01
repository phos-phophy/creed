from typing import Dict, Sequence

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import EvalPrediction


MACRO = "MACRO"
MICRO = "MICRO"
PRECISION = "precision"
RECALL = "recall"
F_SCORE = "f_score"


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

    def save_results(precision, recall, f_score, name):
        score[f"{name} / {PRECISION}"] = precision
        score[f"{name} / {RECALL}"] = recall
        score[f"{name} / {F_SCORE}"] = f_score

    pr, r, f, _ = precision_recall_fscore_support(
        pair_labels_ind, pair_logits_ind, average=None, labels=list(range(num_links)), zero_division=0
    )

    for ind, relation_name in enumerate(relations):
        save_results(pr[ind], r[ind], f[ind], relation_name)

    save_results(np.mean(pr).item(), np.mean(r).item(), np.mean(f).item(), MACRO)

    pr, r, f, _ = precision_recall_fscore_support(
        pair_labels_ind, pair_logits_ind, average='micro', labels=list(range(num_links)), zero_division=0
    )

    save_results(pr, r, f, MICRO)

    return score
