from itertools import chain
from typing import Iterable, List, Optional, Tuple

from .fact import AbstractFact, EntityFact, RelationFact
from .span import Span


class Document:
    def __init__(self, doc_id: str, text: str, sentences: List[Span], facts: Iterable[AbstractFact],
                 coref_chains: Optional[List[Tuple[int]]] = None):
        self._doc_id = doc_id
        self._text = text
        self._sentences = sentences
        self._facts = tuple(facts)
        self._coref_chains = coref_chains
        self._validate_sentences()
        self._validate_facts()
        self._validate_chains()

    @property
    def doc_id(self):
        return self._doc_id

    @property
    def text(self):
        return self._text

    @property
    def sentences(self):
        return self._sentences

    @property
    def facts(self):
        return self._facts

    @property
    def coref_chains(self):
        return self._coref_chains

    def _validate_sentences(self):
        text_span = Span(self.text, 0, len(self.text))

        for ind, sent_span in enumerate(self.sentences):
            if sent_span not in text_span:
                raise ValueError(f"Sentence {ind} should be in the text: {sent_span} not in {text_span}")

    def _validate_chains(self):
        for idx in chain(*self.coref_chains):
            if idx >= len(self.facts):
                raise ValueError(f"There is not {idx}th fact")

    def _validate_facts(self):
        text_span = Span(self.text, 0, len(self.text))

        def validate_entity_fact(abstract_fact: EntityFact):
            if abstract_fact.span not in text_span:
                raise ValueError(f"Span of fact (id={abstract_fact.fact_id}) should be in text: {abstract_fact.span} not in {text_span}")

        for fact in self.facts:
            if isinstance(fact, EntityFact):
                validate_entity_fact(fact)
            elif isinstance(fact, RelationFact):
                validate_entity_fact(fact.from_fact)
                validate_entity_fact(fact.to_fact)
            else:
                raise ValueError

    def add_relation_facts(self, facts: Iterable[RelationFact]):
        self._facts = tuple(list(self._facts) + list(facts))
        self._validate_facts()
