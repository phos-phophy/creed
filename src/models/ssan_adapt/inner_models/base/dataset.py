from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import torch
from src.abstract import AbstractDataset, AbstractFact, Document, EntityFact, FactType, RelationFact, Span


class BaseSSANAdaptDataset(AbstractDataset):
    def __init__(
            self,
            documents: Iterable[Document],
            tokenizer,
            extract_labels: bool,
            evaluation: bool,
            entities: Iterable[str],
            relations: Iterable[str]
    ):
        self._max_ent = self._count_max_ent(documents) if extract_labels and evaluation else None
        self.ent2ind = {ent: ind for ind, ent in enumerate(entities)}
        self.rel2ind = {rel: ind for ind, rel in enumerate(relations)}

        super(BaseSSANAdaptDataset, self).__init__(documents, tokenizer, extract_labels, evaluation)

    @property
    def max_ent(self):
        return self._max_ent

    @staticmethod
    def _count_max_ent(documents: Iterable[Document]):
        def get_ner_count(doc: Document):
            return len(list(filter(lambda fact: fact.fact_type is FactType.ENTITY, doc.facts)))
        doc2ner_count = [1] + list(map(lambda document: get_ner_count(document), documents))
        return max(doc2ner_count)

    def _prepare_document(self, document: Document):
        """
        1) input_ids: LongTensor (len,)
        2) ner_ids: LongTensor  (len,)
        3) dist_ids: LongTensor (max_ent, max_ent)
        4) ent_mask: BoolTensor (max_ent, len)
        5) attention_mask: BoolTensor (len,)
        6) struct_matrix: BoolTensor (5, len, len)
        7) labels: BoolTensor (max_ent, max_ent, num_link)
        8) labels_mask: BoolTensor (max_ent, max_ent)
        """

        if len(document.sentences) == 0:
            return

        # token_to_sentence - map token_ind to the corresponding sentence_ind
        # span_to_token_ind - map span to the corresponding token_ind's
        input_ids, token_to_sentence_ind, span_to_token_ind = self._tokenize(document)

        ner_facts = self._extract_ner_facts(document, span_to_token_ind)

        if len(ner_facts) == 0:
            return

        ner_ids, ent_mask = self._extract_ner_types(ner_facts, span_to_token_ind, input_ids.shape[0])

        struct_matrix = self._extract_struct_matrix(token_to_sentence_ind)

        features = {
            "input_ids": input_ids,
            "ner_ids": ner_ids,
            "dist_ids": ...,
            "ent_mask": ent_mask,
            "attention_mask": torch.ones(input_ids.shape[0]).bool(),
            "struct_matrix": struct_matrix,
            "labels": ...,
            "labels_mask": ...,
        }

        self._documents.append(features)

    def _tokenize(self, document: Document):

        input_ids, token_to_sentence_ind, token_to_span = [], [], []

        for ind, sentence in document.sentences:
            for span in sentence:
                tokens = self.tokenizer(document.get_word(span))[1:-1]  # crop tokens of the beginning and the end

                input_ids.extend(tokens)
                token_to_sentence_ind += [ind] * len(tokens)
                token_to_span += [span] * len(tokens)

        input_ids = input_ids[:self.max_len]
        token_to_sentence_ind = token_to_sentence_ind[:self.max_len]
        token_to_span = token_to_span[:self.max_len]

        span_to_token_ind = defaultdict(list)
        for token_ind, span in enumerate(token_to_span):
            span_to_token_ind[span].append(token_ind)

        return torch.tensor(input_ids, dtype=torch.long), token_to_sentence_ind, span_to_token_ind

    def _extract_ner_facts(self, document: Document, span_to_token_ind: Dict[Span, List[int]]):

        ner_facts = []

        for ind, coref_chain in enumerate(document.coref_chains):
            ner_facts.extend(list(filter(lambda fact: any((span in span_to_token_ind) for span in fact.mentions), coref_chain.facts)))

        return ner_facts[:self.max_ent]

    def _extract_ner_types(self, ner_facts: Tuple[EntityFact, ...], span_to_token_ind: Dict[Span, List[int]], seq_len: int):

        max_ent = self.max_ent if self.max_ent else len(ner_facts)

        ner_ids = torch.zeros(seq_len, dtype=torch.long)
        ent_mask = torch.zeros(seq_len, max_ent, dtype=torch.bool)

        for ind, fact in enumerate(ner_facts):
            fact_type = self.ent2ind[fact.fact_type_id]

            for span in fact.mentions:
                for token_ind in span_to_token_ind.get(span, []):
                    ner_ids[token_ind] = fact_type
                    ent_mask[ind][token_ind] = True

        return ner_ids, ent_mask

    def _extract_struct_matrix(self, token_to_sentence_ind):
        pass
