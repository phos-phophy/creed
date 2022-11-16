from abc import ABCMeta, abstractmethod
from enum import Enum

from span import Span


class FactType(str, Enum):
    ENTITY = "entity"
    RELATION = "relation"


class AbstractFact(metaclass=ABCMeta):
    def __init__(self, fact_id: str, fact_type_id: str, fact_type: FactType, span: Span):
        self._fact_id = fact_id
        self._fact_type_id = fact_type_id
        self._fact_type = fact_type
        self._span = span

    @property
    def fact_id(self):
        return self._fact_id

    @property
    def fact_type_id(self):
        return self._fact_type_id

    @property
    def fact_type(self):
        return self._fact_type

    @property
    def span(self):
        return self._span

    @staticmethod
    @abstractmethod
    def _validate_fact_type(fact_type: FactType):
        pass


class EntityFact(AbstractFact):
    def __init__(self, fact_id: str, fact_type_id: str, fact_type: FactType, span: Span):
        self._validate_fact_type(fact_type)
        super().__init__(fact_id, fact_type_id, fact_type, span)

    @staticmethod
    def _validate_fact_type(fact_type: FactType):
        if fact_type != FactType.ENTITY:
            raise ValueError(f"illegal fact type for entity fact: {fact_type}")


class RelationFact(AbstractFact):
    def __init__(self, fact_id: str, fact_type_id: str, fact_type: FactType, span: Span, from_fact: EntityFact, to_fact: EntityFact):
        self._validate_fact_type(fact_type)

        super().__init__(fact_id, fact_type_id, fact_type, span)
        self._from_fact = from_fact
        self._to_fact = to_fact

    @property
    def from_fact(self):
        return self._from_fact

    @property
    def to_fact(self):
        return self._to_fact

    @staticmethod
    def _validate_fact_type(fact_type: FactType):
        if fact_type != FactType.RELATION:
            raise ValueError(f"illegal fact type for relation fact: {fact_type}")
