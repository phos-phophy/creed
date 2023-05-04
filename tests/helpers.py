from unittest import TestCase

from src.abstract import Document
from torch import Tensor, abs, sum


def equal_docs(case: TestCase, gold_doc: Document, doc: Document):
    case.assertEqual(gold_doc.doc_id, doc.doc_id)
    case.assertEqual(gold_doc.text, doc.text)
    case.assertEqual(gold_doc.words, doc.words)
    case.assertEqual(gold_doc.sentences, doc.sentences)
    case.assertEqual(set(gold_doc.entity_facts), set(doc.entity_facts))
    case.assertEqual(set(gold_doc.relation_facts), set(doc.relation_facts))


def equal_tensors(case: TestCase, gold_tensor: Tensor, tensor: Tensor):
    case.assertEqual(gold_tensor.shape, tensor.shape)
    case.assertAlmostEqual(0, sum(abs(gold_tensor - tensor)).item())
