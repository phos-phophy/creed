import json
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterator, List

from src.abstract import AbstractFact, AbstractLoader, Document, EntityFact, RelationFact, Span


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

        sentences: List[List[Span]] = self._extract_sentences(example)
        text = ' '.join(word for sentence in example['sents'] for word in sentence)

        entity_facts = self._extract_entity_facts(example["vertexSet"], sentences)

        facts: List[AbstractFact] = entity_facts
        facts.extend(self._extract_rel_facts(example.get("labels", []), entity_facts))

        return Document(example["title"], text, sentences, facts)

    @staticmethod
    def _extract_sentences(example: Dict[str, Any]):
        start_idx = 0
        sentences: List[List[Span]] = []

        for sent in example["sents"]:
            sentence = []

            for word in sent:
                sentence.append(Span(start_idx, start_idx + len(word)))
                start_idx += len(word) + 1

            sentences.append(sentence)

        return sentences

    @staticmethod
    def _extract_entity_facts(vertex_set: List[List[Dict]], sentences: List[List[Span]]):

        def get_mention_spans(mention: dict):
            sent_id, start, end = mention["sent_id"], mention["pos"][0], mention["pos"][1]
            return [sentences[sent_id][span_id] for span_id in range(start, end, 1)]

        def build_entity_fact(facts_desc: List[dict], coref_id: int):
            type_id = facts_desc[0]["type"]
            mention_spans = chain.from_iterable(get_mention_spans(mention) for mention in facts_desc)
            return EntityFact("", type_id, coref_id, tuple(set(mention_spans)))

        return [build_entity_fact(coref_facts_desc, ind) for ind, coref_facts_desc in enumerate(vertex_set)]

    @staticmethod
    def _extract_rel_facts(labels: List[Dict], entity_facts: List[EntityFact]):
        return [RelationFact("", desc["r"], entity_facts[desc['h']], entity_facts[desc['t']]) for desc in labels]
