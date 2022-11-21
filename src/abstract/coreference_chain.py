from typing import FrozenSet, Iterable

from .fact import EntityFact


class CoreferenceChain:
    def __init__(self, coref_facts: Iterable[EntityFact]):
        self._facts: FrozenSet[EntityFact] = frozenset(coref_facts)

    @property
    def facts(self):
        return self._facts
