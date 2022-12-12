import json
import unittest
from pathlib import Path

from src.datasets import DocREDConverter
from src.models.ssan_adapt.inner_models.base.dataset import BaseSSANAdaptDataset
from transformers import AutoTokenizer


class BaseSSANAdaptDatasetTest(unittest.TestCase):

    def setUp(self):
        self.converter = DocREDConverter()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        with Path('etc/datasets/docred/rel_info.json').open('r') as file:
            relations = ['<NO_REL>'] + list(json.load(file).items())
            self.relations = tuple(relations)

        with Path('etc/datasets/docred/DocRED_baseline_metadata/ner2id.json').open('r') as file:
            self.entities = tuple(json.load(file).keys())

    def test(self):
        documents = list(self.converter.convert(Path("tests/datasets/examples/docred.json")))
        document = BaseSSANAdaptDataset(documents, self.tokenizer, True, True, self.entities, self.relations, 0, 0)[0]

        expected_shapes = {
            "input_ids": (185,),
            "ner_ids": (185,),
            "dist_ids": (14, 14),
            "ent_mask": (14, 185),
            "attention_mask": (185,),
            "struct_matrix": (5, 185, 185),
            "labels": (14, 14, 97),
            "labels_mask": (14, 14)
        }

        for key, shape in expected_shapes.items():
            self.assertEqual(shape, document[key].shape)
