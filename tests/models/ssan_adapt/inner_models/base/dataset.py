import json
import math
import unittest
from pathlib import Path

from src.abstract import get_tokenizer_len_attribute
from src.datasets import DocREDConverter
from src.models.ssan_adapt.inner_models.base.dataset import BaseSSANAdaptDataset
from transformers import AutoTokenizer


class BaseSSANAdaptDatasetTest(unittest.TestCase):

    def setUp(self):
        self.converter = DocREDConverter()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        self.len_attr = get_tokenizer_len_attribute(self.tokenizer)

        with Path('tests/models/ssan_adapt/data/rel_info.json').open('r') as file:
            relations = ['<NO_REL>'] + list(json.load(file).items())
            self.relations = tuple(relations)

        with Path('tests/models/ssan_adapt/data/ner2id.json').open('r') as file:
            self.entities = tuple(json.load(file).keys())

    def test(self):

        dist_base = 2
        dist_ceil = math.ceil(math.log(self.tokenizer.__getattribute__(self.len_attr), dist_base)) + 1

        documents = list(self.converter.convert(Path("tests/datasets/data/docred.json")))
        document = BaseSSANAdaptDataset(documents, self.tokenizer, True, True, self.entities, self.relations, 0, 0, dist_base, dist_ceil)[0]

        expected_shapes = {
            "features": {
                "input_ids": (185,),
                "ner_ids": (185,),
                "dist_ids": (14, 14),
                "ent_mask": (14, 185),
                "attention_mask": (185,),
                "struct_matrix": (5, 185, 185),
            },
            "labels": {
                "labels": (14, 14, 97),
                "labels_mask": (14, 14)
            }
        }

        for key, shape in expected_shapes["features"].items():
            self.assertEqual(shape, document.features[key].shape)

        for key, shape in expected_shapes["labels"].items():
            self.assertEqual(shape, document.labels[key].shape)
