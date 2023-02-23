from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Tuple

from .span import Span


class FactClass(str, Enum):
    ENTITY = "entity"
    RELATION = "relation"


class AbstractFact(metaclass=ABCMeta):
    def __init__(self, name: str, type_id: str, fact_class: FactClass):
        self._name = name
        self._type_id = type_id
        self._fact_class = fact_class

    def __eq__(self, other: 'AbstractFact'):
        return self.name == other.name and self.type_id == other.type_id and self.fact_class == other.fact_class

    @abstractmethod
    def __hash__(self):
        return hash((self.name, self.type_id, self.fact_class))

    def __repr__(self):
        name, type_id, fact_class = self.name, self.type_id, self.fact_class
        return f"AbstractFact({name=}, {type_id=}, {fact_class=})"

    @property
    def name(self):
        return self._name

    @property
    def type_id(self):
        return self._type_id

    @property
    def fact_class(self):
        return self._fact_class


class EntityFact(AbstractFact):
    def __init__(self, name: str, type_id: str, coreference_id: str, mentions: Tuple[Span, ...]):
        super().__init__(name, type_id, FactClass.ENTITY)
        self._coreference_id = coreference_id
        self._mentions = frozenset(mentions)

    def __eq__(self, other: 'AbstractFact'):
        return isinstance(other, EntityFact) \
            and super().__eq__(other) \
            and self.coreference_id == other.coreference_id \
            and self.mentions == other.mentions

    def __hash__(self):
        return hash((self.name, self.type_id, self.fact_class, self.mentions))

    def __repr__(self):
        name, type_id, fact_class, coreference_id, mentions = self.name, self.type_id, self.fact_class, self.coreference_id, self.mentions
        return f"EntityFact({name=}, {type_id=}, {fact_class=}, {coreference_id=}, {mentions=})"

    @property
    def coreference_id(self):
        return self._coreference_id

    @property
    def mentions(self):
        return self._mentions


class RelationFact(AbstractFact):
    def __init__(self, name: str, type_id: str, from_fact: EntityFact, to_fact: EntityFact):
        super().__init__(name, type_id, FactClass.RELATION)

        self._from_fact = from_fact
        self._to_fact = to_fact

    def __eq__(self, other: 'AbstractFact'):
        return isinstance(other, RelationFact) and super().__eq__(other) and \
            self.from_fact == other.from_fact and self.to_fact == other.to_fact

    def __hash__(self):
        return hash((self.name, self.type_id, self.fact_class, self.from_fact, self.to_fact))

    def __repr__(self):
        name, type_id, fact_class = self.name, self.type_id, self.fact_class
        from_fact_type_id, to_fact_type_id = self.from_fact.type_id, self.to_fact.type_id
        return f"RelationFact({name=}, {type_id=}, {fact_class=}, {from_fact_type_id=}, {to_fact_type_id=})"

    @property
    def from_fact(self):
        return self._from_fact

    @property
    def to_fact(self):
        return self._to_fact
