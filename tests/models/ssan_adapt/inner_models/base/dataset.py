import json
import math
import unittest
from pathlib import Path

import torch
from src.abstract import EntityFact, RelationFact, get_tokenizer_len_attribute
from src.loader import DocREDLoader
from src.models.ssan_adapt.inner_models.base.dataset import BaseSSANAdaptDataset
from tests.helpers import equal_tensors
from transformers import AutoTokenizer


class BaseSSANAdaptDatasetTest(unittest.TestCase):

    def setUp(self):

        with Path('tests/models/ssan_adapt/data/rel_info.json').open('r') as file:
            rel_info = json.load(file)
            relations = ['<NO_REL>'] + list(rel_info.values())
            self.relations = tuple(relations)
            self.rel_to_ind = {rel: ind for ind, rel in enumerate(self.relations)}

        with Path('tests/models/ssan_adapt/data/ner2id.json').open('r') as file:
            self.entities = tuple(json.load(file).keys())

        self.loader = DocREDLoader(rel_info)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        self.len_attr = get_tokenizer_len_attribute(self.tokenizer)

        self.no_ent_ind = 0
        self.no_rel_ind = 0

    def test_shapes(self):

        dist_base = 2
        dist_ceil = math.ceil(math.log(self.tokenizer.__getattribute__(self.len_attr), dist_base)) + 1

        documents = list(self.loader.load(Path("tests/loader/data/docred.json")))
        document = BaseSSANAdaptDataset(
            documents, self.tokenizer, True, True, self.entities, self.relations, self.no_ent_ind, self.no_rel_ind, dist_base, dist_ceil
        )[0]

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

    def test_consistency(self):

        dist_base = 2
        dist_ceil = math.ceil(math.log(self.tokenizer.__getattribute__(self.len_attr), dist_base)) + 1

        documents = list(self.loader.load(Path("tests/loader/data/docred.json")))
        document1 = BaseSSANAdaptDataset(
            documents, self.tokenizer, True, True, self.entities, self.relations, self.no_ent_ind, self.no_rel_ind, dist_base, dist_ceil
        )[0]
        document2 = BaseSSANAdaptDataset(
            documents, self.tokenizer, True, True, self.entities, self.relations, self.no_ent_ind, self.no_rel_ind, dist_base, dist_ceil
        )[0]

        for key, tensor in document1.features.items():
            equal_tensors(self, tensor.long(), document2.features[key].long())

        for key, tensor in document1.labels.items():
            equal_tensors(self, tensor.long(), document2.labels[key].long())

    def test_labels(self):

        dist_base = 2
        dist_ceil = math.ceil(math.log(self.tokenizer.__getattribute__(self.len_attr), dist_base)) + 1

        documents = list(self.loader.load(Path("tests/loader/data/docred.json")))
        document = BaseSSANAdaptDataset(
            documents, self.tokenizer, True, True, self.entities, self.relations, self.no_ent_ind, self.no_rel_ind, dist_base, dist_ceil
        )[0]

        rel_facts = tuple(filter(lambda fact: isinstance(fact, RelationFact) and fact.type_id in self.relations, documents[0].facts))
        ent_facts = tuple(filter(lambda fact: isinstance(fact, EntityFact), documents[0].facts))
        fact_to_ind = {fact: ind for ind, fact in enumerate(ent_facts)}

        expected_labels = torch.zeros(14, 14, 97, dtype=torch.bool)
        expected_labels_mask = torch.ones(14, 14, dtype=torch.bool)

        expected_labels[:, :, self.no_rel_ind] = True
        for rel_fact in rel_facts:

            if rel_fact.from_fact not in ent_facts or rel_fact.to_fact not in ent_facts:
                continue

            source_fact_ind = fact_to_ind[rel_fact.from_fact]
            target_fact_ind = fact_to_ind[rel_fact.to_fact]

            expected_labels[source_fact_ind][target_fact_ind][self.rel_to_ind[rel_fact.type_id]] = True
            expected_labels[source_fact_ind][target_fact_ind][self.no_rel_ind] = False

        equal_tensors(self, expected_labels.long(), document.labels["labels"].long())
        equal_tensors(self, expected_labels_mask.long(), document.labels["labels_mask"].long())

    def test_labels_num(self):
        dist_base = 2
        dist_ceil = math.ceil(math.log(self.tokenizer.__getattribute__(self.len_attr), dist_base)) + 1

        documents = list(self.loader.load(Path("tests/loader/data/docred.json")))

        gold_document = documents[0]
        pred_document = BaseSSANAdaptDataset(
            documents, self.tokenizer, True, True, self.entities, self.relations, self.no_ent_ind, self.no_rel_ind, dist_base, dist_ceil
        )[0]

        gold_labels_num = len(list(filter(lambda fact: isinstance(fact, RelationFact), gold_document.facts)))

        labels: torch.Tensor = pred_document.labels["labels"]
        pred_labels_num = torch.sum(labels[:, :, 1:], dtype=torch.int).item()  # without NO_REL link

        self.assertEqual(gold_labels_num, pred_labels_num)
