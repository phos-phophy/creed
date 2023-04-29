import unittest
from pathlib import Path

from src.abstract import Document, EntityFact, Mention, RelationFact, Word
from src.loader import DocREDLoader
from tests.helpers import equal_docs


class DocREDLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.loader = DocREDLoader()

    def test_simple_example(self):
        example = {
            "title": "title",
            "sents": [["First", "sentence", '.'], ["End", "of", "text", '!']],
            "vertexSet": [
                [{"pos": [0, 1], "sent_id": 0, "type": "NUM", "name": "First"}],
                [{"pos": [0, 1], "sent_id": 1, "type": "POS", "name": "End"}],
                [{"pos": [2, 3], "sent_id": 1, "type": "MISC", "name": "text"}]
            ],
            "labels": [{"r": "P17", "h": 1, "t": 2, "evidence": []}]
        }

        doc_id = example["title"]
        sentences = [[Word('First', 0, 0, 0), Word('sentence', 0, 1, 1), Word('.', 0, 2, 2)],
                     [Word('End', 1, 0, 3), Word('of', 1, 1, 4), Word('text', 1, 2, 5), Word('!', 1, 3, 6)]]
        facts = [
            EntityFact("", "NUM", 0, (Mention([Word('First', 0, 0, 0)]),)),
            EntityFact("", "POS", 1, (Mention([Word('End', 1, 0, 3)]),)),
            EntityFact("", "MISC", 2, (Mention([Word('text', 1, 2, 5)]),))
        ]
        facts.extend([RelationFact("", "P17", facts[1], facts[2])])

        gold_doc = Document(doc_id, sentences, tuple(facts))

        document = self.loader._build_document(example)
        equal_docs(self, gold_doc, document)

    def test_coreference_facts(self):
        example = {
            "title": "title",
            "sents": [["First", "sentence", '.'], ["End", "of", "text", '!']],
            "vertexSet": [
                [{"pos": [0, 1], "sent_id": 0, "type": "NUM", "name": "First"}],
                [{"pos": [0, 1], "sent_id": 1, "type": "POS", "name": "End"}],
                [{"pos": [1, 2], "sent_id": 1, "type": "MISC", "name": "of"}, {"pos": [2, 3], "sent_id": 1, "type": "MISC", "name": "text"}]
            ],
            "labels": [{"r": "P17", "h": 1, "t": 2, "evidence": []}]
        }

        doc_id = example["title"]
        sentences = [[Word('First', 0, 0, 0), Word('sentence', 0, 1, 1), Word('.', 0, 2, 2)],
                     [Word('End', 1, 0, 3), Word('of', 1, 1, 4), Word('text', 1, 2, 5), Word('!', 1, 3, 6)]]
        facts = [
            EntityFact("", "NUM", 0, (Mention([Word('First', 0, 0, 0)]),)),
            EntityFact("", "POS", 1, (Mention([Word('End', 1, 0, 3)]),)),
            EntityFact("", "MISC", 2, (Mention([Word('of', 1, 1, 4)]), Mention([Word('text', 1, 2, 5)]),))
        ]
        facts.extend([RelationFact("", "P17", facts[1], facts[2])])

        gold_doc = Document(doc_id, sentences, tuple(facts))

        document = self.loader._build_document(example)
        equal_docs(self, gold_doc, document)

    def test_identical_coreference_facts(self):
        example = {
            "title": "title",
            "sents": [["First", "sentence", '.'], ["End", "of", "text", '!']],
            "vertexSet": [
                [{"pos": [0, 1], "sent_id": 0, "type": "NUM", "name": "First"}],
                [{"pos": [0, 1], "sent_id": 1, "type": "POS", "name": "End"}],
                [{"pos": [1, 2], "sent_id": 1, "type": "MISC", "name": "of"}, {"pos": [1, 2], "sent_id": 1, "type": "MISC", "name": "of"}]
            ],
            "labels": [{"r": "P17", "h": 1, "t": 2, "evidence": []}]
        }

        doc_id = example["title"]
        sentences = [[Word('First', 0, 0, 0), Word('sentence', 0, 1, 1), Word('.', 0, 2, 2)],
                     [Word('End', 1, 0, 3), Word('of', 1, 1, 4), Word('text', 1, 2, 5), Word('!', 1, 3, 6)]]
        facts = [
            EntityFact("", "NUM", 0, (Mention([Word('First', 0, 0, 0)]),)),
            EntityFact("", "POS", 1, (Mention([Word('End', 1, 0, 3)]),)),
            EntityFact("", "MISC", 2, (Mention([Word('of', 1, 1, 4)]),))
        ]
        facts.extend([RelationFact("", "P17", facts[1], facts[2])])

        gold_doc = Document(doc_id, sentences, tuple(facts))

        document = self.loader._build_document(example)
        equal_docs(self, gold_doc, document)

    def test_real_example(self):
        doc_id = "Skai TV"
        sentences = [[Word("Skai", 0, 0, 0), Word("TV", 0, 1, 1), Word("is", 0, 2, 2), Word("a", 0, 3, 3), Word("Greek", 0, 4, 4),
                      Word("free", 0, 5, 5), Word("-", 0, 6, 6), Word("to", 0, 7, 7), Word("-", 0, 8, 8), Word("air", 0, 9, 9),
                      Word("television", 0, 10, 10), Word("network", 0, 11, 11), Word("based", 0, 12, 12), Word("in", 0, 13, 13),
                      Word("Piraeus", 0, 14, 14), Word(".", 0, 15, 15)],
                     [Word("It", 1, 0, 16), Word("is", 1, 1, 17), Word("part", 1, 2, 18), Word("of", 1, 3, 19), Word("the", 1, 4, 20),
                      Word("Skai", 1, 5, 21), Word("Group", 1, 6, 22), Word(",", 1, 7, 23), Word("one", 1, 8, 24), Word("of", 1, 9, 25),
                      Word("the", 1, 10, 26), Word("largest", 1, 11, 27), Word("media", 1, 12, 28), Word("groups", 1, 13, 29),
                      Word("in", 1, 14, 30), Word("the", 1, 15, 31), Word("country", 1, 16, 32), Word(".", 1, 17, 33)],
                     [Word("It", 2, 0, 34), Word("was", 2, 1, 35), Word("relaunched", 2, 2, 36), Word("in", 2, 3, 37),
                      Word("its", 2, 4, 38), Word("present", 2, 5, 39), Word("form", 2, 6, 40), Word("on", 2, 7, 41), Word("1st", 2, 8, 42),
                      Word("of", 2, 9, 43), Word("April", 2, 10, 44), Word("2006", 2, 11, 45), Word("in", 2, 12, 46),
                      Word("the", 2, 13, 47), Word("Athens", 2, 14, 48), Word("metropolitan", 2, 15, 49), Word("area", 2, 16, 50),
                      Word(",", 2, 17, 51), Word("and", 2, 18, 52), Word("gradually", 2, 19, 53), Word("spread", 2, 20, 54),
                      Word("its", 2, 21, 55), Word("coverage", 2, 22, 56), Word("nationwide", 2, 23, 57), Word(".", 2, 24, 58)],
                     [Word("Besides", 3, 0, 59), Word("digital", 3, 1, 60), Word("terrestrial", 3, 2, 61), Word("transmission", 3, 3, 62),
                      Word(",", 3, 4, 63), Word("it", 3, 5, 64), Word("is", 3, 6, 65), Word("available", 3, 7, 66), Word("on", 3, 8, 67),
                      Word("the", 3, 9, 68), Word("subscription", 3, 10, 69), Word("-", 3, 11, 70), Word("based", 3, 12, 71),
                      Word("encrypted", 3, 13, 72), Word("services", 3, 14, 73), Word("of", 3, 15, 74), Word("Nova", 3, 16, 75),
                      Word("and", 3, 17, 76), Word("Cosmote", 3, 18, 77), Word("TV", 3, 19, 78), Word(".", 3, 20, 79)],
                     [Word("Skai", 4, 0, 80), Word("TV", 4, 1, 81), Word("is", 4, 2, 82), Word("also", 4, 3, 83), Word("a", 4, 4, 84),
                      Word("member", 4, 5, 85), Word("of", 4, 6, 86), Word("Digea", 4, 7, 87), Word(",", 4, 8, 88), Word("a", 4, 9, 89),
                      Word("consortium", 4, 10, 90), Word("of", 4, 11, 91), Word("private", 4, 12, 92), Word("television", 4, 13, 93),
                      Word("networks", 4, 14, 94), Word("introducing", 4, 15, 95), Word("digital", 4, 16, 96),
                      Word("terrestrial", 4, 17, 97), Word("transmission", 4, 18, 98), Word("in", 4, 19, 99), Word("Greece", 4, 20, 100),
                      Word(".", 4, 21, 101)],
                     [Word("At", 5, 0, 102), Word("launch", 5, 1, 103), Word(",", 5, 2, 104), Word("Skai", 5, 3, 105),
                      Word("TV", 5, 4, 106), Word("opted", 5, 5, 107), Word("for", 5, 6, 108), Word("dubbing", 5, 7, 109),
                      Word("all", 5, 8, 110), Word("foreign", 5, 9, 111), Word("language", 5, 10, 112), Word("content", 5, 11, 113),
                      Word("into", 5, 12, 114), Word("Greek", 5, 13, 115), Word(",", 5, 14, 116), Word("instead", 5, 15, 117),
                      Word("of", 5, 16, 118), Word("using", 5, 17, 119), Word("subtitles", 5, 18, 120), Word(".", 5, 19, 121)],
                     [Word("This", 6, 0, 122), Word("is", 6, 1, 123), Word("very", 6, 2, 124), Word("uncommon", 6, 3, 125),
                      Word("in", 6, 4, 126), Word("Greece", 6, 5, 127), Word("for", 6, 6, 128), Word("anything", 6, 7, 129),
                      Word("except", 6, 8, 130), Word("documentaries", 6, 9, 131), Word("(", 6, 10, 132), Word("using", 6, 11, 133),
                      Word("voiceover", 6, 12, 134), Word("dubbing", 6, 13, 135), Word(")", 6, 14, 136), Word("and", 6, 15, 137),
                      Word("children", 6, 16, 138), Word("'s", 6, 17, 139), Word("programmes", 6, 18, 140), Word("(", 6, 19, 141),
                      Word("using", 6, 20, 142), Word("lip", 6, 21, 143), Word("-", 6, 22, 144), Word("synced", 6, 23, 145),
                      Word("dubbing", 6, 24, 146), Word(")", 6, 25, 147), Word(",", 6, 26, 148), Word("so", 6, 27, 149),
                      Word("after", 6, 28, 150), Word("intense", 6, 29, 151), Word("criticism", 6, 30, 152), Word("the", 6, 31, 153),
                      Word("station", 6, 32, 154), Word("switched", 6, 33, 155), Word("to", 6, 34, 156), Word("using", 6, 35, 157),
                      Word("subtitles", 6, 36, 158), Word("for", 6, 37, 159), Word("almost", 6, 38, 160), Word("all", 6, 39, 161),
                      Word("foreign", 6, 40, 162), Word("shows", 6, 41, 163), Word(".", 6, 42, 164)]
                     ]

        facts = [
            EntityFact("", "ORG", 0, (Mention([Word("Skai", 0, 0, 0), Word("TV", 0, 1, 1)]),
                                      Mention([Word("Skai", 4, 0, 80), Word("TV", 4, 1, 81)]),
                                      Mention([Word("Skai", 5, 3, 105), Word("TV", 5, 4, 106)]))),
            EntityFact("", "LOC", 1, (Mention([Word("Greek", 0, 4, 4)]),)),
            EntityFact("", "LOC", 2, (Mention([Word("Piraeus", 0, 14, 14)]),)),
            EntityFact("", "ORG", 3, (Mention([Word("Skai", 1, 5, 21), Word("Group", 1, 6, 22)]),)),
            EntityFact("", "TIME", 4, (Mention([
                Word("1st", 2, 8, 42), Word("of", 2, 9, 43), Word("April", 2, 10, 44), Word("2006", 2, 11, 45)
            ]),)),
            EntityFact("", "LOC", 5, (Mention([Word("Athens", 2, 14, 48)]),)),
            EntityFact("", "ORG", 6, (Mention([Word("Nova", 3, 16, 75)]),)),
            EntityFact("", "ORG", 7, (Mention([Word("Cosmote", 3, 18, 77), Word("TV", 3, 19, 78)]),)),
            EntityFact("", "ORG", 8, (Mention([Word("Digea", 4, 7, 87)]),)),
            EntityFact("", "LOC", 9, (Mention([Word("Greece", 4, 20, 100)]), Mention([Word("Greece", 6, 5, 127)]))),
            EntityFact("", "MISC", 10, (Mention([Word("Greek", 5, 13, 115)]),))
        ]

        facts.extend([
            RelationFact("", "P17", facts[2], facts[9]),
            RelationFact("", "P17", facts[3], facts[9]),
            RelationFact("", "P17", facts[5], facts[9]),
            RelationFact("", "P159", facts[0], facts[2]),
            RelationFact("", "P127", facts[0], facts[3]),
            RelationFact("", "P159", facts[0], facts[5]),
            RelationFact("", "P17", facts[0], facts[9]),

        ])

        gold_document = Document(doc_id, sentences, tuple(facts))

        document = list(self.loader.load(Path("tests/loader/data/docred_1.json")))[0]
        equal_docs(self, gold_document, document)
