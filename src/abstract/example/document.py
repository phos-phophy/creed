from collections import defaultdict
from itertools import chain
from typing import Dict, Iterable, Tuple

from src.abstract.feature import (
    AbstractFact,
    EntityFact,
    FactClass,
    Mention,
    RelationFact,
    Word
)


class Document:
    """ A base class that represents one example from a dataset """

    def __init__(self, doc_id: str, sentences: Iterable[Iterable[Word]], facts: Iterable[AbstractFact]):

        self._doc_id = doc_id
        self._sentences = tuple(tuple(sentence) for sentence in sentences)
        self._facts = tuple(facts)

        self._validate_facts()

        self._coreference_chains = self._build_coreference_chains(facts)

    @property
    def doc_id(self):
        return self._doc_id

    @property
    def text(self):
        return ' '.join(word.text for word in self.words)

    @property
    def words(self):
        return tuple(chain.from_iterable(self.sentences))

    @property
    def sentences(self):
        return self._sentences

    @property
    def facts(self):
        return self._facts

    @property
    def coreference_chains(self):
        return self._coreference_chains

    @staticmethod
    def _build_coreference_chains(facts) -> Dict[int, Tuple[EntityFact]]:
        coreference_chains = defaultdict(list)
        for fact in facts:
            if fact.fact_class is FactClass.ENTITY:
                fact: EntityFact
                coreference_chains[fact.coreference_id].append(fact)
        return {key: tuple(item) for key, item in coreference_chains.items()}

    def _validate_mentions(self, mentions: Iterable[Mention]):
        words = self.words
        for mention in mentions:
            for word in mention.words:
                if word not in words:
                    raise ValueError(f'Word "{word}" is not in text!')

    def _validate_facts(self):
        for fact in self.facts:
            if isinstance(fact, EntityFact):
                self._validate_mentions(fact.mentions)
            elif isinstance(fact, RelationFact):
                self._validate_mentions(fact.from_fact.mentions)
                self._validate_mentions(fact.to_fact.mentions)
            else:
                raise ValueError

    def add_relation_facts(self, facts: Iterable[RelationFact]):
        self._facts = tuple(list(self._facts) + list(facts))
        self._validate_facts()
