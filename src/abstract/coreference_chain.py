from typing import FrozenSet, Iterable

from .fact import EntityFact


class CoreferenceChain:
    def __init__(self, coref_facts: Iterable[EntityFact]):
        self._facts: FrozenSet[EntityFact] = frozenset(coref_facts)

    def __eq__(self, other: 'CoreferenceChain'):
        return self._facts == other._facts

    def __hash__(self):
        return hash(self.facts)

    def __repr__(self):
        return f"Coreference chain: {len(self.facts)} fact(s), {self.facts[0].fact_type}, {self.facts[0].fact_type_id}"

    @property
    def facts(self):
        return tuple(self._facts)
