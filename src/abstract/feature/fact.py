from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Tuple

from .span import Span


class FactType(str, Enum):
    ENTITY = "entity"
    RELATION = "relation"


class AbstractFact(metaclass=ABCMeta):
    def __init__(self, fact_id: str, fact_type_id: str, fact_type: FactType):
        self._fact_id = fact_id
        self._fact_type_id = fact_type_id
        self._fact_type = fact_type

    def __eq__(self, other: 'AbstractFact'):
        return self.fact_id == other.fact_id and self.fact_type_id == other.fact_type_id and self.fact_type == other.fact_type

    @abstractmethod
    def __hash__(self):
        return hash((self.fact_id, self.fact_type_id, self.fact_type))

    def __repr__(self):
        return f"{self.fact_id=}, {self.fact_type_id=}, {self.fact_type=}"

    @property
    def fact_id(self):
        return self._fact_id

    @property
    def fact_type_id(self):
        return self._fact_type_id

    @property
    def fact_type(self):
        return self._fact_type


class EntityFact(AbstractFact):
    def __init__(self, fact_id: str, fact_type_id: str, coreference_id: str, mentions: Tuple[Span, ...]):
        super().__init__(fact_id, fact_type_id, FactType.ENTITY)
        self._coreference_id = coreference_id
        self._mentions = mentions
        self._validate_mentions()

    def __eq__(self, other: 'AbstractFact'):
        return isinstance(other, EntityFact) and super().__eq__(other) and self.mentions == other.mentions

    def __hash__(self):
        return hash((self.fact_id, self.fact_type_id, self.fact_type, self.mentions))

    @property
    def coreference_id(self):
        return self._coreference_id

    @property
    def mentions(self):
        return self._mentions

    def _validate_mentions(self):
        if len(self.mentions) != len(set(self.mentions)):
            raise ValueError(f"EntityFact ({self}) has identical mentions!")


class RelationFact(AbstractFact):
    def __init__(self, fact_id: str, fact_type_id: str, from_fact: EntityFact, to_fact: EntityFact):
        super().__init__(fact_id, fact_type_id, FactType.RELATION)

        self._from_fact = from_fact
        self._to_fact = to_fact

    def __eq__(self, other: 'AbstractFact'):
        return isinstance(other, RelationFact) and super().__eq__(other) and \
            self.from_fact == other.from_fact and self.to_fact == other.to_fact

    def __hash__(self):
        return hash((self.fact_id, self.fact_type_id, self.fact_type, self.from_fact, self.to_fact))

    @property
    def from_fact(self):
        return self._from_fact

    @property
    def to_fact(self):
        return self._to_fact
