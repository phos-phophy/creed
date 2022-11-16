from abc import ABCMeta, abstractmethod
from enum import Enum

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

    @property
    def fact_id(self):
        return self._fact_id

    @property
    def fact_type_id(self):
        return self._fact_type_id

    @property
    def fact_type(self):
        return self._fact_type

    @abstractmethod
    def _validate_fact_type(self):
        pass


class EntityFact(AbstractFact):
    def __init__(self, fact_id: str, fact_type_id: str, fact_type: FactType, span: Span):
        super().__init__(fact_id, fact_type_id, fact_type)
        self._span = span
        self._validate_fact_type()

    def __eq__(self, other: 'AbstractFact'):
        return isinstance(other, EntityFact) and super().__eq__(other) and self.span == other.span

    @property
    def span(self):
        return self._span

    def _validate_fact_type(self):
        if self.fact_type != FactType.ENTITY:
            raise ValueError(f"illegal fact type for entity fact: {self.fact_type}")


class RelationFact(AbstractFact):
    def __init__(self, fact_id: str, fact_type_id: str, fact_type: FactType, from_fact: EntityFact, to_fact: EntityFact):
        super().__init__(fact_id, fact_type_id, fact_type)
        self._validate_fact_type()

        self._from_fact = from_fact
        self._to_fact = to_fact

    def __eq__(self, other: 'AbstractFact'):
        return isinstance(other, RelationFact) and super().__eq__(other) and \
            self.from_fact == other.from_fact and self.to_fact == other.to_fact

    @property
    def from_fact(self):
        return self._from_fact

    @property
    def to_fact(self):
        return self._to_fact

    def _validate_fact_type(self):
        if self.fact_type != FactType.RELATION:
            raise ValueError(f"illegal fact type for relation fact: {self.fact_type}")
