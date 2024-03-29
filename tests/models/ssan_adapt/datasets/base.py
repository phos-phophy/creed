import json
import math
import unittest
from pathlib import Path

import torch
from src.abstract import DiversifierConfig, NO_REL_IND, get_tokenizer_len_attribute
from src.loader import DocREDLoader
from src.models.ssan_adapt.datasets import BaseDataset
from tests.helpers import equal_tensors
from transformers import AutoTokenizer

from .base_gold import GoldDataset


class BaseSSANAdaptDatasetTest(unittest.TestCase):

    def setUp(self):

        with Path('tests/models/ssan_adapt/data/rel_info.json').open('r') as file:
            rel_info = json.load(file)
            relations = ['<NO_REL>'] + list(rel_info.keys())
            self.relations = tuple(relations)
            self.rel_to_ind = {rel: ind for ind, rel in enumerate(self.relations)}

        with Path('tests/models/ssan_adapt/data/ner2id.json').open('r') as file:
            self.entities = tuple(json.load(file).keys())

        self.loader = DocREDLoader()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        self.len_attr = get_tokenizer_len_attribute(self.tokenizer)

        self.diversifier = DiversifierConfig()

    def test_shapes(self):

        dist_base = 2
        dist_ceil = math.ceil(math.log(self.tokenizer.__getattribute__(self.len_attr), dist_base)) + 1

        documents = list(self.loader.load(Path("tests/loader/data/docred_1.json")))
        document = BaseDataset(
            documents, self.tokenizer, True, True, self.entities, self.relations, dist_base, dist_ceil, '', self.diversifier
        ).prepare_documents()[0]

        expected_shapes = {
            "features": {
                "input_ids": (187,),
                "ner_ids": (187,),
                "dist_ids": (11, 11),
                "ent_mask": (11, 187),
                "attention_mask": (187,),
                "struct_matrix": (5, 187, 187),
            },
            "labels": {
                "labels": (11, 11, 97),
                "labels_mask": (11, 11)
            }
        }

        for key, shape in expected_shapes["features"].items():
            self.assertEqual(shape, document.features[key].shape)

        for key, shape in expected_shapes["labels"].items():
            self.assertEqual(shape, document.labels[key].shape)

    def test_consistency_1(self):

        dist_base = 2
        dist_ceil = math.ceil(math.log(self.tokenizer.__getattribute__(self.len_attr), dist_base)) + 1

        documents = list(self.loader.load(Path("tests/loader/data/docred_1.json")))
        document1 = BaseDataset(
            documents, self.tokenizer, True, True, self.entities, self.relations, dist_base, dist_ceil, '', self.diversifier
        ).prepare_documents()[0]
        document2 = BaseDataset(
            documents, self.tokenizer, True, True, self.entities, self.relations, dist_base, dist_ceil, '', self.diversifier
        ).prepare_documents()[0]

        for key, tensor in document1.features.items():
            equal_tensors(self, tensor.long(), document2.features[key].long())

        for key, tensor in document1.labels.items():
            equal_tensors(self, tensor.long(), document2.labels[key].long())

    def test_consistency_2(self):

        dist_base = 2
        dist_ceil = math.ceil(math.log(self.tokenizer.__getattribute__(self.len_attr), dist_base)) + 1

        documents = list(self.loader.load(Path("tests/loader/data/docred_1.json")))
        documents = documents + documents
        documents = BaseDataset(
            documents, self.tokenizer, True, True, self.entities, self.relations, dist_base, dist_ceil, '', self.diversifier
        ).prepare_documents()

        document1 = documents[0]
        document2 = documents[1]

        for key, tensor in document1.features.items():
            equal_tensors(self, tensor.long(), document2.features[key].long())

        for key, tensor in document1.labels.items():
            equal_tensors(self, tensor.long(), document2.labels[key].long())

    def test_labels(self):

        dist_base = 2
        dist_ceil = math.ceil(math.log(self.tokenizer.__getattribute__(self.len_attr), dist_base)) + 1

        documents = list(self.loader.load(Path("tests/loader/data/docred_1.json")))
        document = BaseDataset(
            documents, self.tokenizer, True, True, self.entities, self.relations, dist_base, dist_ceil, '', self.diversifier
        ).prepare_documents()[0]

        rel_facts = tuple(filter(lambda fact: fact.type_id in self.relations, documents[0].relation_facts))
        ent_facts = documents[0].entity_facts
        fact_to_ind = {fact: ind for ind, fact in enumerate(ent_facts)}

        expected_labels = torch.zeros(11, 11, 97, dtype=torch.bool)
        expected_labels_mask = torch.ones(11, 11, dtype=torch.bool)

        expected_labels[:, :, NO_REL_IND] = True
        for rel_fact in rel_facts:

            if rel_fact.from_fact not in ent_facts or rel_fact.to_fact not in ent_facts:
                continue

            source_fact_ind = fact_to_ind[rel_fact.from_fact]
            target_fact_ind = fact_to_ind[rel_fact.to_fact]

            expected_labels[source_fact_ind][target_fact_ind][self.rel_to_ind[rel_fact.type_id]] = True
            expected_labels[source_fact_ind][target_fact_ind][NO_REL_IND] = False

        equal_tensors(self, expected_labels.long(), document.labels["labels"].long())
        equal_tensors(self, expected_labels_mask.long(), document.labels["labels_mask"].long())

    def test_labels_num(self):
        dist_base = 2
        dist_ceil = math.ceil(math.log(self.tokenizer.__getattribute__(self.len_attr), dist_base)) + 1

        documents = list(self.loader.load(Path("tests/loader/data/docred_1.json")))

        gold_document = documents[0]
        pred_document = BaseDataset(
            documents, self.tokenizer, True, True, self.entities, self.relations, dist_base, dist_ceil, '', self.diversifier
        ).prepare_documents()[0]

        gold_labels_num = len(gold_document.relation_facts)

        labels: torch.Tensor = pred_document.labels["labels"]
        labels = torch.cat((labels[:, :, :NO_REL_IND], labels[:, :, NO_REL_IND + 1:]), dim=-1)  # without NO_REL link
        pred_labels_num = torch.sum(labels, dtype=torch.int).item()

        self.assertEqual(gold_labels_num, pred_labels_num)

    def test_struct_matrix(self):
        dist_base = 2
        dist_ceil = math.ceil(math.log(self.tokenizer.__getattribute__(self.len_attr), dist_base)) + 1

        documents = list(self.loader.load(Path("tests/loader/data/docred_200.json")))

        pred_dataset = BaseDataset(
            documents, self.tokenizer, True, True, self.entities, self.relations, dist_base, dist_ceil, '', self.diversifier
        ).prepare_documents()
        gold_dataset = GoldDataset(
            documents, self.tokenizer, True, True, self.entities, self.relations, dist_base, dist_ceil, '', self.diversifier
        ).prepare_documents()

        for p_doc, g_doc in zip(pred_dataset, gold_dataset):
            equal_tensors(self, p_doc.features["struct_matrix"] * 1, g_doc.features["struct_matrix"] * 1)
