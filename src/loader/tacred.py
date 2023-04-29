import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

from src.abstract import AbstractFact, AbstractLoader, Document, EntityFact, Mention, RelationFact, Word


"""
Example from TACRED consists of the following fields:
1) id: str - unique hash code of the current example
2) docid: str - the LDC document id from which this example was drawn (an id that can be found in the LDC2018T03 source corpus)
3) relation: str - relations type
4) token: List[str] - list of the words
5) subj_start: int - index of subject's start token
6) subj_end: int - index of subject's end token
7) obj_start: int - index of object's start token
8) obj_end: int -index of object's end token
9) subj_type: str - entity type of subject
10) obj_type: str - entity type of object
11) stanford_pos: List[str] - list of part-of-speech tags of the words
12) stanford_ner: List[str] - list of named entity tags of the words
13) stanford_deprel: List[str] - list of dependency relations of the words to its head tokens
14) stanford_head: List[str] - list of 1-based indexes of the dependency head of the words
"""


class TacredLoader(AbstractLoader):

    def load(self, path: Path) -> Iterator[Document]:
        with path.open('r') as file:
            examples = json.load(file)

        return map(self._build_document, examples)

    def _build_document(self, example: Dict[str, Any]) -> Document:

        sentences: List[List[Word]] = self._extract_sentences(example)

        entity_facts: List[EntityFact] = self._extract_entity_facts(example, sentences)

        facts: List[AbstractFact] = entity_facts
        if example["relation"] != "no_relation":
            facts.append(RelationFact("", example["relation"], entity_facts[0], entity_facts[1]))

        return Document(example["docid"], sentences, facts)

    @staticmethod
    def _extract_sentences(example: Dict[str, Any]):
        return [[Word(word, 0, ind, ind) for ind, word in enumerate(example["token"])]]

    @staticmethod
    def _extract_entity_facts(example: Dict[str, Any], sentences: List[List[Word]]):

        def get_mention(start, end):
            return Mention(sentences[0][span_id] for span_id in range(start, end + 1, 1))

        facts = [
            EntityFact("subject", example["subj_type"], 0, (get_mention(example["subj_start"], example["subj_end"]),)),
            EntityFact("object", example["obj_type"], 1, (get_mention(example["obj_start"], example["obj_end"]),))
        ]

        return facts
