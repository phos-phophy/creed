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

WARNING: There are erroneous examples in DocRED in which the same coreference etities (from vertexSet) are used several times.
But there are also documents with the nested entities that are ok (in the case they have identical types)

E.g. in the ../docred/train_annotated.json file there are:
* 'Nabacalis' document where second fact is embedded in the first one - it's ok
* 'Eclipse Aviation' document where 9th set of coreference facts contains 2 identical facts (about Vern Raburn) - it is not ok
"""


class DocREDLoader(AbstractLoader):

    def __init__(self, rel_info: Dict[str, str] = None):
        self._rel_info = rel_info if rel_info else {}

    @property
    def rel_info(self):
        return self._rel_info

    def load(self, path: Path) -> Iterator[Document]:
        with path.open('r') as file:
            examples = json.load(file)

        return map(self._build_document, examples)

    def _build_document(self, example: Dict[str, Any]) -> Document:

        sentences: List[List[Span]] = self._extract_sentences(example)
        text = ' '.join(word for sentence in example['sents'] for word in sentence)

        entity_facts, coref_chains = self._extract_entity_facts(example["vertexSet"], sentences)

        facts: List[AbstractFact] = entity_facts
        facts.extend(self._extract_rel_facts(example.get("labels", []), coref_chains))

        return Document(example["title"], text, sentences, facts)

    @staticmethod
    def _extract_sentences(example: Dict):
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

        facts: List[EntityFact] = []
        coref_chains: List[List[EntityFact]] = []

        def build_entity_fact(desc: dict, coref_id: int):
            mention_spans = [sentences[desc["sent_id"]][span_id] for span_id in range(desc["pos"][0], desc["pos"][1], 1)]
            return EntityFact("", desc["type"], str(coref_id), tuple(mention_spans))

        for ind, coref_facts_desc in enumerate(vertex_set):
            coref_facts = {build_entity_fact(fact_desc, ind) for fact_desc in coref_facts_desc}
            facts.extend(coref_facts)
            coref_chains.append(list(coref_facts))

        return facts, coref_chains

    def _extract_rel_facts(self, labels: List[Dict], coref_facts: List[List[EntityFact]]):

        def build_rel_fact(desc: dict):
            rel_type = self.rel_info.get(desc["r"], desc["r"])
            from_facts = coref_facts[desc["h"]]
            to_facts = coref_facts[desc["t"]]
            return [RelationFact("", rel_type, from_fact, to_fact) for from_fact in from_facts for to_fact in to_facts]

        return list(chain.from_iterable(build_rel_fact(rel_desc) for rel_desc in labels))
