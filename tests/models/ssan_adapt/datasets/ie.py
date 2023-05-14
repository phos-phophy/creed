import json
import math
import unittest
from pathlib import Path

from src.abstract import DiversifierConfig, Document, EntityFact, Mention, Word, get_tokenizer_len_attribute
from src.loader import DocREDLoader
from src.models.ssan_adapt.datasets import IETypesDataset
from transformers import AutoTokenizer


class IETypesDatasetTest(unittest.TestCase):

    def setUp(self):
        with Path('tests/models/ssan_adapt/data/rel_info.json').open('r') as file:
            rel_info = json.load(file)
            relations = ['<NO_REL>'] + list(rel_info.keys())
            self.relations = tuple(relations)
            self.rel_to_ind = {rel: ind for ind, rel in enumerate(self.relations)}

        self.entities = ('NO_ENT', 'ENT')

        self.loader = DocREDLoader()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        self.len_attr = get_tokenizer_len_attribute(self.tokenizer)

        self.sentences = [
            [Word("word", 0, 0, 0), Word("word", 0, 1, 1), Word("ent", 0, 2, 2), Word(".", 0, 3, 3)],
            [Word("ent", 1, 0, 4), Word("word", 1, 1, 5), Word("ent", 1, 2, 6), Word("ent", 1, 3, 7), Word(".", 1, 4, 8)],
            [Word("word", 2, 0, 9), Word("word", 2, 1, 10), Word(".", 2, 2, 11)],
            [Word("word", 3, 0, 12), Word("ent", 3, 1, 13), Word("word", 3, 2, 14), Word("word", 3, 3, 15), Word(".", 3, 4, 16)]
        ]

    def test_get_ent_tokens_1(self):
        dist_base = 2
        dist_ceil = math.ceil(math.log(self.tokenizer.__getattribute__(self.len_attr), dist_base)) + 1

        diversifier = DiversifierConfig()

        facts = (EntityFact('', 'ENT1', 1, (Mention([Word("ent", 0, 2, 2)]),)),
                 EntityFact('', 'ENT1', 2, (Mention([Word("ent", 1, 0, 4)]),)),
                 EntityFact('', 'ENT2', 3, (Mention([Word("ent", 1, 2, 6), Word("ent", 1, 3, 7)]), Mention([Word("ent", 3, 1, 13)]))))

        document = Document('', self.sentences, facts)
        dataset = IETypesDataset(
            [document], self.tokenizer, True, True, self.entities, self.relations, dist_base, dist_ceil, '', diversifier
        ).prepare_documents()

        start_ent_tokens, end_ent_tokens = dataset._get_ent_tokens(document)

        expected_start = ['', '', ' <ENT1> ', '', ' <ENT1> ', '', ' <ENT2> ', '', '', '', '', '', '', ' <ENT2> ', '', '', '']
        expected_end = ['', '', ' </ENT1> ', '', ' </ENT1> ', '', '', ' </ENT2> ', '', '', '', '', '', ' </ENT2> ', '', '', '']

        self.assertEqual(expected_start, start_ent_tokens)
        self.assertEqual(expected_end, end_ent_tokens)

    def test_get_ent_tokens_2(self):
        dist_base = 2
        dist_ceil = math.ceil(math.log(self.tokenizer.__getattribute__(self.len_attr), dist_base)) + 1

        diversifier = DiversifierConfig()

        facts = (EntityFact('', 'ENT1', 1, (Mention([Word("ent", 0, 2, 2)]),)),
                 EntityFact('', 'ENT1', 2, (Mention([Word("ent", 1, 0, 4)]),)),
                 EntityFact('', 'ENT2', 4, (Mention([Word("ent", 1, 2, 6)]),)),
                 EntityFact('', 'ENT3', 3, (Mention([Word("ent", 1, 3, 7)]), Mention([Word("ent", 3, 1, 13)]))))

        document = Document('', self.sentences, facts)
        dataset = IETypesDataset(
            [document], self.tokenizer, True, True, self.entities, self.relations, dist_base, dist_ceil, '', diversifier
        ).prepare_documents()

        start_ent_tokens, end_ent_tokens = dataset._get_ent_tokens(document)

        expected_start = ['', '', ' <ENT1> ', '', ' <ENT1> ', '', ' <ENT2> ', ' <ENT3> ', '', '', '', '', '', ' <ENT3> ', '', '', '']
        expected_end = ['', '', ' </ENT1> ', '', ' </ENT1> ', '', ' </ENT2> ', ' </ENT3> ', '', '', '', '', '', ' </ENT3> ', '', '', '']

        self.assertEqual(expected_start, start_ent_tokens)
        self.assertEqual(expected_end, end_ent_tokens)

    def test_diversifier(self):
        dist_base = 2
        dist_ceil = math.ceil(math.log(self.tokenizer.__getattribute__(self.len_attr), dist_base)) + 1

        diversifier = DiversifierConfig(replace_prob=1, mapping={'ENT2': ['ENT3']})

        facts = (EntityFact('', 'ENT1', 1, (Mention([Word("ent", 0, 2, 2)]),)),
                 EntityFact('', 'ENT1', 2, (Mention([Word("ent", 1, 0, 4)]),)),
                 EntityFact('', 'ENT2', 3, (Mention([Word("ent", 1, 2, 6), Word("ent", 1, 3, 7)]), Mention([Word("ent", 3, 1, 13)]))))

        document = Document('', self.sentences, facts)
        dataset = IETypesDataset(
            [document], self.tokenizer, True, True, self.entities, self.relations, dist_base, dist_ceil, '', diversifier
        ).prepare_documents()

        start_ent_tokens, end_ent_tokens = dataset._get_ent_tokens(document)

        expected_start = ['', '', ' <ENT1> ', '', ' <ENT1> ', '', ' <ENT3> ', '', '', '', '', '', '', ' <ENT3> ', '', '', '']
        expected_end = ['', '', ' </ENT1> ', '', ' </ENT1> ', '', '', ' </ENT3> ', '', '', '', '', '', ' </ENT3> ', '', '', '']

        self.assertEqual(expected_start, start_ent_tokens)
        self.assertEqual(expected_end, end_ent_tokens)
