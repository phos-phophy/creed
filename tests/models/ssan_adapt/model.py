import json
import unittest
from pathlib import Path

import torch
from src.abstract import ModelScore, Score
from src.models import SSANAdaptModel


class SSANAdaptModelTest(unittest.TestCase):
    def setUp(self) -> None:

        with Path('tests/models/ssan_adapt/data/ner2id.json').open('r') as file:
            entities = tuple(json.load(file).keys())

        config = {'entities': entities,
                  'relations': ['1', '2'],
                  'inner_model_type': 'base',
                  'hidden_dim': 256,
                  'dropout': 0.1,
                  'no_ent_ind': 0,
                  'dist_base': 2,
                  'pretrained_model_path': 'bert-base-uncased',
                  'tokenizer_path': 'bert-base-uncased'}

        self.model: SSANAdaptModel = SSANAdaptModel(**config)

    def test_compute_loss(self):
        pass

    def test_score(self):

        logits = torch.tensor([[
            [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]],  # 3 0 2
            [[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]],  # 2 2 1
            [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],  # 5 0 0
            [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],  # 3 1 1
            [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]]   # 4 1 0
        ]])  # (1, 5, 5, 3)

        gold_labels = {
            "labels": torch.tensor([
                [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]],  # 4 0 1
                [[0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]],  # 2 3 0
                [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],  # 5 0 0
                [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],  # 5 0 0
                [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]]   # 4 0 1
            ],
                dtype=torch.float),  # (5, 5, 3)
            "labels_mask": torch.ones(5, 5)
        }

        # 1'st relation: 17 pt, 0 fp, 3 fn; precision = 1, recall = 17 / 20
        # 2'nd relation: 2 pt, 2 fp, 1 fn; precision = 0.5, recall = 2 / 3
        # 3'rd relation: 1 pt, 3 fp, 1 fn; precision = 0.25, recall = 0.5
        # micro: 20 pt, 5 fp, 5 fn; precision = 0.8, recall = 0.8
        # macro: precision = 1.75 / 3, recall = (17 / 20 + 2 / 3 + 0.5) / 3

        score_1 = Score(precision=1, recall=17 / 20, f_score=2 * 17 / 20 / (1 + 17 / 20))
        score_2 = Score(precision=0.5, recall=2 / 3, f_score=2 * (0.5 * 2 / 3) / (0.5 + 2 / 3))
        score_3 = Score(precision=0.25, recall=0.5, f_score=2 * (0.25 * 0.5) / (0.25 + 0.5))

        gold_micro_score = Score(precision=0.8, recall=0.8, f_score=2 * 0.8 * 0.8 / 1.6)
        gold_macro_score = Score(
            precision=(score_1.precision + score_2.precision + score_3.precision) / 3,
            recall=(score_1.recall + score_2.recall + score_3.recall) / 3,
            f_score=(score_1.f_score + score_2.f_score + score_3.f_score) / 3
        )
        gold_relations_score = {
            self.model.relations[0]: score_1,
            self.model.relations[1]: score_2,
            self.model.relations[2]: score_3
        }

        model_score: ModelScore = self.model.score([logits], [gold_labels])

        self.assertEqual(gold_macro_score, model_score.macro_score)
        self.assertEqual(gold_micro_score, model_score.micro_score)
        self.assertDictEqual(gold_relations_score, model_score.relations_score)
