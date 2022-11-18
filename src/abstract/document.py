from itertools import chain
from typing import Iterable, Optional, Tuple

from .fact import AbstractFact, EntityFact, RelationFact
from .span import Span


class Document:
    def __init__(self, doc_id: str, text: str, words: Tuple[Span], sentences: Tuple[Span], facts: Iterable[AbstractFact],
                 coref_chains: Optional[Tuple[Tuple[int]]] = None):

        self._doc_id = doc_id
        self._text = text
        self._words = words
        self._sentences = sentences
        self._facts = tuple(facts)
        self._coref_chains = coref_chains

        self._validate_spans(words)
        self._validate_spans(sentences)
        self._validate_facts()
        self._validate_chains()

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
    def coref_chains(self):
        return self._coref_chains

    @staticmethod
    def _validate_span(text_span: Span, span: Span):
        if span not in text_span:
            raise ValueError(f"Span should be in the text: {span} not in {text_span}")

    def _validate_spans(self, spans: Tuple[Span]):
        text_span = Span(0, len(self.text))

        for span in spans:
            self._validate_span(text_span, span)

    def _validate_chains(self):
        for idx in chain(*self.coref_chains):
            if idx >= len(self.facts):
                raise ValueError(f"There is not {idx}th fact")

    def _validate_facts(self):
        text_span = Span(0, len(self.text))

        for fact in self.facts:
            if isinstance(fact, EntityFact):
                self._validate_span(text_span, fact.span)
            elif isinstance(fact, RelationFact):
                self._validate_span(text_span, fact.from_fact.span)
                self._validate_span(text_span, fact.to_fact.span)
            else:
                raise ValueError

    def get_word(self, span: Span):
        if span in self.words:
            return self.text[span.start_idx: span.end_idx]
        raise ValueError(f"Document does not contain a word in the span: {span}")

    def add_relation_facts(self, facts: Iterable[RelationFact]):
        self._facts = tuple(list(self._facts) + list(facts))
        self._validate_facts()
