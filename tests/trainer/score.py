import unittest
from typing import Tuple

import numpy as np
import torch
from src.trainer.score import F_SCORE, MACRO, MICRO, PRECISION, RECALL, score_model
from transformers import EvalPrediction


class ScoreModelTest(unittest.TestCase):

    def test_score(self):

        relations = ['zero', 'one', 'two']

        logits = torch.tensor([[
            [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]],  # 3 0 2
            [[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]],  # 2 2 1
            [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],  # 5 0 0
            [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],  # 3 1 1
            [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]]   # 4 1 0
        ]])  # (1, 5, 5, 3)

        labels = torch.tensor([[
            [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]],  # 4 0 1
            [[0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]],  # 2 3 0
            [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],  # 5 0 0
            [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],  # 5 0 0
            [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]]   # 4 0 1
        ]],
            dtype=torch.float)  # (1, 5, 5, 3)

        labels_mask = torch.ones(5, 5)

        label_ids: Tuple[np.ndarray, ...] = (labels.numpy(), labels_mask.numpy())
        eval_prediction = EvalPrediction(predictions=logits.numpy(), label_ids=label_ids)
        model_score: dict = score_model(eval_prediction, relations)

        # 1'st relation: 17 pt, 0 fp, 3 fn; precision = 1, recall = 17 / 20
        # 2'nd relation: 2 pt, 2 fp, 1 fn; precision = 0.5, recall = 2 / 3
        # 3'rd relation: 1 pt, 3 fp, 1 fn; precision = 0.25, recall = 0.5
        # micro: 20 pt, 5 fp, 5 fn; precision = 0.8, recall = 0.8
        # macro: precision = 1.75 / 3, recall = (17 / 20 + 2 / 3 + 0.5) / 3

        gold_score = {
            f"{relations[0]} / {PRECISION}": 1,
            f"{relations[0]} / {RECALL}": 17 / 20,
            f"{relations[0]} / {F_SCORE}": 2 * 17 / 20 / (1 + 17 / 20),

            f"{relations[1]} / {PRECISION}": 0.5,
            f"{relations[1]} / {RECALL}": 2 / 3,
            f"{relations[1]} / {F_SCORE}": 2 * (0.5 * 2 / 3) / (0.5 + 2 / 3),

            f"{relations[2]} / {PRECISION}": 0.25,
            f"{relations[2]} / {RECALL}": 0.5,
            f"{relations[2]} / {F_SCORE}": 2 * (0.25 * 0.5) / (0.25 + 0.5),

            f"{MICRO} / {PRECISION}": 0.8,
            f"{MICRO} / {RECALL}": 0.8,
            f"{MICRO} / {F_SCORE}": 2 * 0.8 * 0.8 / (0.8 + 0.8),
        }

        def macro(name):
            gold_score[f"{MACRO} / {name}"] = (gold_score[f"{relations[0]} / {name}"]
                                               + gold_score[f"{relations[1]} / {name}"]
                                               + gold_score[f"{relations[2]} / {name}"]) / 3

        macro(PRECISION), macro(RECALL), macro(F_SCORE)

        self.assertDictEqual(gold_score, model_score)
