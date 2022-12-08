from unittest import TestCase

from src.abstract import Document


def equal_docs(case: TestCase, gold_doc: Document, doc: Document):
    case.assertEqual(gold_doc.doc_id, doc.doc_id)
    case.assertEqual(gold_doc.text, doc.text)
    case.assertEqual(gold_doc.words, doc.words)
    case.assertEqual(gold_doc.sentences, doc.sentences)
    case.assertEqual(set(gold_doc.facts), set(doc.facts))
