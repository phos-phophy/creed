import unittest
from pathlib import Path

from src.abstract import Document, EntityFact, Mention, RelationFact, Word
from src.loader import TacredLoader
from tests.helpers import equal_docs


class TacredLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.loader = TacredLoader()

    def test_without_rel(self):
        doc_id = "eng-NG-31-101172-8859554"

        sentences = [[Word("He", 0, 0, 0), Word("has", 0, 1, 1), Word("served", 0, 2, 2), Word("as", 0, 3, 3), Word("a", 0, 4, 4),
                      Word("policy", 0, 5, 5), Word("aide", 0, 6, 6), Word("to", 0, 7, 7), Word("the", 0, 8, 8), Word("late", 0, 9, 9),
                      Word("U.S.", 0, 10, 10), Word("Senator", 0, 11, 11), Word("Alan", 0, 12, 12), Word("Cranston", 0, 13, 13),
                      Word(",", 0, 14, 14), Word("as", 0, 15, 15), Word("National", 0, 16, 16), Word("Issues", 0, 17, 17),
                      Word("Director", 0, 18, 18), Word("for", 0, 19, 19), Word("the", 0, 20, 20), Word("2004", 0, 21, 21),
                      Word("presidential", 0, 22, 22), Word("campaign", 0, 23, 23), Word("of", 0, 24, 24), Word("Congressman", 0, 25, 25),
                      Word("Dennis", 0, 26, 26), Word("Kucinich", 0, 27, 27), Word(",", 0, 28, 28), Word("as", 0, 29, 29),
                      Word("a", 0, 30, 30), Word("co-founder", 0, 31, 31), Word("of", 0, 32, 32), Word("Progressive", 0, 33, 33),
                      Word("Democrats", 0, 34, 34), Word("of", 0, 35, 35), Word("America", 0, 36, 36), Word("and", 0, 37, 37),
                      Word("as", 0, 38, 38), Word("a", 0, 39, 39), Word("member", 0, 40, 40), Word("of", 0, 41, 41), Word("the", 0, 42, 42),
                      Word("international", 0, 43, 43), Word("policy", 0, 44, 44), Word("department", 0, 45, 45), Word("at", 0, 46, 46),
                      Word("the", 0, 47, 47), Word("RAND", 0, 48, 48), Word("Corporation", 0, 49, 49), Word("think", 0, 50, 50),
                      Word("tank", 0, 51, 51), Word("before", 0, 52, 52), Word("all", 0, 53, 53), Word("that", 0, 54, 54),
                      Word(".", 0, 55, 55)]]

        facts = [
            EntityFact("subject", "ORGANIZATION", 0, (Mention([
                Word("Progressive", 0, 33, 33), Word("Democrats", 0, 34, 34), Word("of", 0, 35, 35), Word("America", 0, 36, 36)
            ]),)),
            EntityFact("object", "ORGANIZATION", 1, (Mention([
                Word("international", 0, 43, 43), Word("policy", 0, 44, 44), Word("department", 0, 45, 45)
            ]),))
        ]

        gold_document = Document(doc_id, sentences, facts)

        document = list(self.loader.load(Path('tests/loader/data/tacred_without_rel.json')))[0]
        equal_docs(self, gold_document, document)

    def test_with_rel(self):
        doc_id = "APW_ENG_20101103.0539"

        sentences = [[Word("At", 0, 0, 0), Word("the", 0, 1, 1), Word("same", 0, 2, 2), Word("time", 0, 3, 3), Word(",", 0, 4, 4),
                      Word("Chief", 0, 5, 5), Word("Financial", 0, 6, 6), Word("Officer", 0, 7, 7), Word("Douglas", 0, 8, 8),
                      Word("Flint", 0, 9, 9), Word("will", 0, 10, 10), Word("become", 0, 11, 11), Word("chairman", 0, 12, 12),
                      Word(",", 0, 13, 13), Word("succeeding", 0, 14, 14), Word("Stephen", 0, 15, 15), Word("Green", 0, 16, 16),
                      Word("who", 0, 17, 17), Word("is", 0, 18, 18), Word("leaving", 0, 19, 19), Word("to", 0, 20, 20),
                      Word("take", 0, 21, 21), Word("a", 0, 22, 22), Word("government", 0, 23, 23), Word("job", 0, 24, 24),
                      Word(".", 0, 25, 25)]]

        facts = [EntityFact("subject", "PERSON", 0, (Mention([Word("Douglas", 0, 8, 8), Word("Flint", 0, 9, 9)]),)),
                 EntityFact("object", "TITLE", 1, (Mention([Word("chairman", 0, 12, 12)]),))]

        facts.extend([RelationFact("", "per:title", facts[0], facts[1])])

        gold_document = Document(doc_id, sentences, facts)

        document = list(self.loader.load(Path('tests/loader/data/tacred_with_rel.json')))[0]
        equal_docs(self, gold_document, document)
