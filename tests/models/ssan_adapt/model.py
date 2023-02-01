import json
import unittest
from pathlib import Path

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
