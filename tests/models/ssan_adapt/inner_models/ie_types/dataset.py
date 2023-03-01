import json
import math
import unittest
from pathlib import Path

from src.abstract import Document, EntityFact, Span, get_tokenizer_len_attribute
from src.loader import DocREDLoader
from src.models.ssan_adapt.inner_models.ie_types.dataset import IETypesSSANAdaptDataset
from transformers import AutoTokenizer


class IETypesSSANAdaptDatasetTest(unittest.TestCase):

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

    def test_get_ent_tokens_1(self):
        dist_base = 2
        dist_ceil = math.ceil(math.log(self.tokenizer.__getattribute__(self.len_attr), dist_base)) + 1

        text = "word word ent. ent word ent ent. word word. word ent word word."
        sentences = ((Span(0, 4), Span(5, 9), Span(10, 13), Span(13, 14)),
                     (Span(15, 18), Span(19, 23), Span(24, 27), Span(28, 31), Span(31, 32)),
                     (Span(33, 37), Span(38, 42), Span(42, 43)),
                     (Span(44, 48), Span(49, 52), Span(53, 57), Span(58, 62), Span(62, 63)))

        facts = (EntityFact('', 'ENT1', '1', (Span(10, 13),)),
                 EntityFact('', 'ENT1', '2', (Span(15, 18),)),
                 EntityFact('', 'ENT2', '3', (Span(24, 27), Span(28, 31), Span(49, 52))))

        document = Document('', text, sentences, facts)
        dataset = IETypesSSANAdaptDataset([document], self.tokenizer, True, True, self.entities, self.relations, dist_base, dist_ceil, '')

        start_ent_tokens, end_ent_tokens = dataset._get_ent_tokens(document)

        expected_start = ['', '', ' <ENT1> ', '', ' <ENT1> ', '', ' <ENT2> ', '', '', '', '', '', '', ' <ENT2> ', '', '', '']
        expected_end = ['', '', ' </ENT1> ', '', ' </ENT1> ', '', '', ' </ENT2> ', '', '', '', '', '', ' </ENT2> ', '', '', '']

        self.assertEqual(expected_start, start_ent_tokens)
        self.assertEqual(expected_end, end_ent_tokens)

    def test_get_ent_tokens_2(self):
        dist_base = 2
        dist_ceil = math.ceil(math.log(self.tokenizer.__getattribute__(self.len_attr), dist_base)) + 1

        text = "word word ent. ent word ent ent. word word. word ent word word."
        sentences = ((Span(0, 4), Span(5, 9), Span(10, 13), Span(13, 14)),
                     (Span(15, 18), Span(19, 23), Span(24, 27), Span(28, 31), Span(31, 32)),
                     (Span(33, 37), Span(38, 42), Span(42, 43)),
                     (Span(44, 48), Span(49, 52), Span(53, 57), Span(58, 62), Span(62, 63)))

        facts = (EntityFact('', 'ENT1', '1', (Span(10, 13),)),
                 EntityFact('', 'ENT1', '2', (Span(15, 18),)),
                 EntityFact('', 'ENT2', '4', (Span(24, 27),)),
                 EntityFact('', 'ENT3', '3', (Span(28, 31), Span(49, 52))))

        document = Document('', text, sentences, facts)
        dataset = IETypesSSANAdaptDataset([document], self.tokenizer, True, True, self.entities, self.relations, dist_base, dist_ceil, '')

        start_ent_tokens, end_ent_tokens = dataset._get_ent_tokens(document)

        expected_start = ['', '', ' <ENT1> ', '', ' <ENT1> ', '', ' <ENT2> ', ' <ENT3> ', '', '', '', '', '', ' <ENT3> ', '', '', '']
        expected_end = ['', '', ' </ENT1> ', '', ' </ENT1> ', '', ' </ENT2> ', ' </ENT3> ', '', '', '', '', '', ' </ENT3> ', '', '', '']

        self.assertEqual(expected_start, start_ent_tokens)
        self.assertEqual(expected_end, end_ent_tokens)
