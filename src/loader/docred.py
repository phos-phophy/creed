import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

from src.abstract import AbstractLoader, Document, EntityFact, Mention, RelationFact, Word


"""
Example from DocRED consists of the following fields:
1) title: str
2) sents: List[List[str]] - list of lists of words
3) vertexSet: List[List[Dict]] - list of entity's mentions, each mention is dict with following fields:
    * name: str
    * pos: Tuple[int, int]
    * sent_id: int
    * type: str
4) labels: List[dict] - list of relations (test file does not have labels)
    * r: str - type of relation
    * h: int - head (source) entity index in vertexSet
    * t: int - tail (target) entity index in vertexSet
    * evidence:  List[int] - ???
"""


class DocREDLoader(AbstractLoader):

    def load(self, path: Path) -> Iterator[Document]:
        with path.open('r') as file:
            examples = json.load(file)

        return map(self._build_document, examples)

    def _build_document(self, example: Dict[str, Any]) -> Document:

        sentences: List[List[Word]] = self._extract_sentences(example)

        entity_facts = self._extract_entity_facts(example["vertexSet"], sentences)
        relation_facts = self._extract_rel_facts(example.get("labels", []), entity_facts)

        return Document(example["title"], sentences, entity_facts, relation_facts)

    @staticmethod
    def _extract_sentences(example: Dict[str, Any]):
        sentences: List[List[Word]] = []

        ind_in_doc = 0
        for sent_ind, sent in enumerate(example["sents"]):
            sentence = []

            for ind_in_sent, word in enumerate(sent):
                sentence.append(Word(word, sent_ind, ind_in_sent, ind_in_doc))
                ind_in_doc += 1

            sentences.append(sentence)

        return sentences

    @staticmethod
    def _extract_entity_facts(vertex_set: List[List[Dict]], sentences: List[List[Word]]):

        def get_mention_spans(mention: dict):
            sent_id, start, end = mention["sent_id"], mention["pos"][0], mention["pos"][1]
            return Mention(sentences[sent_id][span_id] for span_id in range(start, end, 1))

        def build_entity_fact(facts_desc: List[dict], coref_id: int):
            type_id = facts_desc[0]["type"]
            mention_spans = {get_mention_spans(mention) for mention in facts_desc}
            return EntityFact("", type_id, coref_id, tuple(mention_spans))

        return [build_entity_fact(coref_facts_desc, ind) for ind, coref_facts_desc in enumerate(vertex_set)]

    @staticmethod
    def _extract_rel_facts(labels: List[Dict], entity_facts: List[EntityFact]):
        return [RelationFact("", desc["r"], entity_facts[desc['h']], entity_facts[desc['t']]) for desc in labels]
