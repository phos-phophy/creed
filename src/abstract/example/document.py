from collections import defaultdict
from itertools import chain
from typing import Dict, Iterable, Tuple

from src.abstract.feature import EntityFact, RelationFact, Word


class Document:
    """ A base class that represents one example from a dataset """

    def __init__(
            self,
            doc_id: str,
            sentences: Iterable[Iterable[Word]],
            entity_facts: Iterable[EntityFact],
            relation_facts: Iterable[RelationFact] = None
    ):

        self._doc_id = doc_id
        self._sentences = tuple(tuple(sentence) for sentence in sentences)
        self._entity_facts = tuple(entity_facts)
        self._relation_facts = tuple(relation_facts) if relation_facts else tuple()

        self._coreference_chains = self._build_coreference_chains(self._entity_facts)

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
    def entity_facts(self):
        return self._entity_facts

    @property
    def relation_facts(self):
        return self._relation_facts

    @property
    def coreference_chains(self):
        return self._coreference_chains

    @staticmethod
    def _build_coreference_chains(facts: Iterable[EntityFact]) -> Dict[int, Tuple[EntityFact]]:
        coreference_chains = defaultdict(list)
        for fact in facts:
            coreference_chains[fact.coreference_id].append(fact)
        return {key: tuple(item) for key, item in coreference_chains.items()}

    def add_relation_facts(self, facts: Iterable[RelationFact]):
        self._relation_facts = tuple(list(self._relation_facts) + list(facts))
