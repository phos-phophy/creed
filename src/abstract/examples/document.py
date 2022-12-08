from collections import defaultdict
from itertools import chain
from typing import Iterable, Tuple

from src.abstract.features import (
    AbstractFact,
    EntityFact,
    FactType,
    RelationFact,
    Span
)


class Document:
    def __init__(self, doc_id: str, text: str, sentences: Iterable[Iterable[Span]], facts: Iterable[AbstractFact]):

        self._doc_id = doc_id
        self._text = text
        self._words = tuple(chain.from_iterable(sentences))
        self._sentences = tuple(tuple(sentence) for sentence in sentences)
        self._facts = tuple(facts)

        self._validate_spans(self._words)
        self._validate_facts()

        self._coreference_chains = self._build_coreference_chains(facts)

    @property
    def doc_id(self):
        return self._doc_id

    @property
    def text(self):
        return self._text

    @property
    def words(self):
        return self._words

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
    def _build_coreference_chains(facts):
        coreference_chains = defaultdict(list)
        for fact in facts:
            if fact.fact_type is FactType.ENTITY:
                fact: EntityFact
                coreference_chains[fact.coreference_id].append(fact)
        return coreference_chains

    @staticmethod
    def _validate_span(text_span: Span, span: Span):
        if span not in text_span:
            raise ValueError(f"Span should be in the text: {span} not in {text_span}")

    def _validate_spans(self, spans: Tuple[Span, ...]):
        text_span = Span(0, len(self.text))

        for span in spans:
            self._validate_span(text_span, span)

    def _validate_facts(self):
        for fact in self.facts:
            if isinstance(fact, EntityFact):
                self._validate_spans(fact.mentions)
            elif isinstance(fact, RelationFact):
                self._validate_spans(fact.from_fact.mentions)
                self._validate_spans(fact.to_fact.mentions)
            else:
                raise ValueError

    def get_word(self, span: Span):
        if span in self.words:
            return self.text[span.start_idx: span.end_idx]
        raise ValueError(f"Document does not contain a word in the span: {span}")

    def add_relation_facts(self, facts: Iterable[RelationFact]):
        self._facts = tuple(list(self._facts) + list(facts))
        self._validate_facts()
