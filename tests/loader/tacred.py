import unittest
from pathlib import Path

from src.abstract import Document, EntityFact, RelationFact, Span
from src.loader import TacredLoader
from tests.helpers import equal_docs


class TacredLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.loader = TacredLoader()

    def test_without_rel(self):

        doc_id = "eng-NG-31-101172-8859554"
        text = 'He has served as a policy aide to the late U.S. Senator Alan Cranston , as National Issues Director for the 2004 ' \
               'presidential campaign of Congressman Dennis Kucinich , as a co-founder of Progressive Democrats of America and as ' \
               'a member of the international policy department at the RAND Corporation think tank before all that .'

        sentences = [[Span(0, 2), Span(3, 6), Span(7, 13), Span(14, 16), Span(17, 18), Span(19, 25), Span(26, 30), Span(31, 33),
                      Span(34, 37), Span(38, 42), Span(43, 47), Span(48, 55), Span(56, 60), Span(61, 69), Span(70, 71), Span(72, 74),
                      Span(75, 83), Span(84, 90), Span(91, 99), Span(100, 103), Span(104, 107), Span(108, 112), Span(113, 125),
                      Span(126, 134), Span(135, 137), Span(138, 149), Span(150, 156), Span(157, 165), Span(166, 167), Span(168, 170),
                      Span(171, 172), Span(173, 183), Span(184, 186), Span(187, 198), Span(199, 208), Span(209, 211), Span(212, 219),
                      Span(220, 223), Span(224, 226), Span(227, 228), Span(229, 235), Span(236, 238), Span(239, 242), Span(243, 256),
                      Span(257, 263), Span(264, 274), Span(275, 277), Span(278, 281), Span(282, 286), Span(287, 298), Span(299, 304),
                      Span(305, 309), Span(310, 316), Span(317, 320), Span(321, 325), Span(326, 327)]]

        facts = [EntityFact("", "ORGANIZATION", 0, (Span(187, 198), Span(199, 208), Span(209, 211), Span(212, 219))),
                 EntityFact("", "ORGANIZATION", 1, (Span(243, 256), Span(257, 263), Span(264, 274)))]

        gold_document = Document(doc_id, text, sentences, facts)

        document = list(self.loader.load(Path('tests/loader/data/tacred_without_rel.json')))[0]
        equal_docs(self, gold_document, document)

    def test_with_rel(self):

        doc_id = "APW_ENG_20101103.0539"
        text = 'At the same time , Chief Financial Officer Douglas Flint will become chairman , succeeding Stephen Green who is leaving ' \
               'to take a government job .'

        sentences = [[Span(0, 2), Span(3, 6), Span(7, 11), Span(12, 16), Span(17, 18), Span(19, 24), Span(25, 34), Span(35, 42),
                     Span(43, 50), Span(51, 56), Span(57, 61), Span(62, 68), Span(69, 77), Span(78, 79), Span(80, 90), Span(91, 98),
                     Span(99, 104), Span(105, 108), Span(109, 111), Span(112, 119), Span(120, 122), Span(123, 127), Span(128, 129),
                     Span(130, 140), Span(141, 144), Span(145, 146)]]

        facts = [EntityFact("", "PERSON", 0, (Span(43, 50), Span(51, 56))),
                 EntityFact("", "TITLE", 1, (Span(69, 77),))]

        facts.extend([RelationFact("", "per:title", facts[0], facts[1])])

        gold_document = Document(doc_id, text, sentences, facts)

        document = list(self.loader.load(Path('tests/loader/data/tacred_with_rel.json')))[0]
        equal_docs(self, gold_document, document)
