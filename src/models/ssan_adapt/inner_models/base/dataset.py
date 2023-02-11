from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from src.abstract import AbstractDataset, Document, EntityFact, FactType, PreparedDocument, RelationFact, Span, get_tokenizer_len_attribute


class BaseSSANAdaptDataset(AbstractDataset):
    def __init__(
            self,
            documents: Iterable[Document],
            tokenizer,
            extract_labels: bool,
            evaluation: bool,
            entities: Iterable[str],
            relations: Iterable[str],
            no_ent_ind: int,
            no_rel_ind: int,
            dist_base: int,
            dist_ceil: int
    ):
        self._max_ent = self._count_max_ent(documents) if (extract_labels and evaluation) else None
        self._entities = tuple(entities)
        self._relations = tuple(relations)
        self._ent_to_ind = {ent: ind for ind, ent in enumerate(entities)}
        self._rel_to_ind = {rel: ind for ind, rel in enumerate(relations)}

        self._no_ent_ind = no_ent_ind
        self._no_rel_ind = no_rel_ind

        self._usual_token = "<USUAL_TOKEN>"

        self._len_attr = get_tokenizer_len_attribute(tokenizer)
        self._distance_encoder = self._init_distance_encoder(tokenizer.__getattribute__(self._len_attr))

        self._dist_bins = torch.tensor([dist_base ** i for i in range(dist_ceil)], dtype=torch.long)

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

    @staticmethod
    def _init_distance_encoder(max_len):
        encoder = torch.zeros(max_len, dtype=torch.long)
        ind = 1
        while ind < max_len:
            encoder[ind:] += 1
            ind <<= 1
        return encoder

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

        # token_to_coreference_id - map token_ind to the coreference_id of the corresponding fact
        ner_ids, ent_mask, token_to_coreference_id = self._extract_ner_types(ner_facts, span_to_token_ind, input_ids.shape[0])

        features = {
            "input_ids": input_ids,
            "ner_ids": ner_ids,
            "dist_ids": self._extract_dist_ids(ent_mask),
            "ent_mask": ent_mask,
            "attention_mask": torch.ones(input_ids.shape[0]).bool(),
            "struct_matrix": self._extract_struct_matrix(token_to_sentence_ind, token_to_coreference_id),
        }

        labels = None
        if self.extract_labels:
            labels_tensors, labels_mask = self._extract_labels_and_mask(ner_facts, self._extract_link_facts(document))
            labels = {"labels": labels_tensors, "labels_mask": labels_mask}

        prepared_document = PreparedDocument(features=features, labels=labels)

        self._documents.append(prepared_document)

    def _tokenize(self, document: Document):

        input_ids, token_to_sentence_ind, token_to_span = [], [], []

        for ind, sentence in enumerate(document.sentences):
            for span in sentence:
                tokens = self.tokenizer(document.get_word(span))['input_ids'][1:-1]  # crop tokens of the beginning and the end

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
        ent_facts = tuple(filter(lambda fact: fact.fact_type is FactType.ENTITY, document.facts))
        return tuple(filter(lambda fact: any((span in span_to_token_ind) for span in fact.mentions), ent_facts))[:self.max_ent]

    @staticmethod
    def _extract_link_facts(document: Document):
        return tuple(filter(lambda fact: fact.fact_type is FactType.RELATION, document.facts))

    def _extract_ner_types(self, ner_facts: Tuple[EntityFact, ...], span_to_token_ind: Dict[Span, List[int]], seq_len: int):

        max_ent = self.max_ent if self.max_ent else len(ner_facts)

        ner_ids = torch.zeros(seq_len, dtype=torch.long)
        ent_mask = torch.zeros(max_ent, seq_len, dtype=torch.bool)
        token_to_coreference_id = [self._usual_token] * seq_len

        for ind, fact in enumerate(ner_facts):
            fact_type = self._ent_to_ind[fact.fact_type_id]

            for span in fact.mentions:
                for token_ind in span_to_token_ind.get(span, []):
                    ner_ids[token_ind] = fact_type
                    ent_mask[ind][token_ind] = True
                    token_to_coreference_id[token_ind] = fact.coreference_id

        # normalization
        tmp = ent_mask.sum(dim=-1)
        tmp = tmp + (tmp == 0)
        ent_mask = ent_mask / tmp.unsqueeze(1)

        return ner_ids, ent_mask, token_to_coreference_id

    def _extract_struct_matrix(self, token_to_sentence_ind: List[int], token_to_coreference_id: List[str]):
        length = len(token_to_sentence_ind)
        struct_mask = torch.zeros((5, length, length), dtype=torch.bool)

        for i in range(length):

            if token_to_coreference_id[i] == self._usual_token:
                continue

            for j in range(length):

                if token_to_sentence_ind[i] != token_to_sentence_ind[j]:
                    if token_to_coreference_id[i] == token_to_coreference_id[j]:
                        struct_mask[0][i][j] = True  # inter-coref
                    elif token_to_coreference_id[j] != self._usual_token:
                        struct_mask[1][i][j] = True  # inter-relate
                else:
                    if token_to_coreference_id[i] == token_to_coreference_id[j]:
                        struct_mask[2][i][j] = True  # intra-coref
                    elif token_to_coreference_id[j] != self._usual_token:
                        struct_mask[3][i][j] = True  # intra-relate
                    else:
                        struct_mask[4][i][j] = True  # intra-NA

        return struct_mask

    def _extract_dist_ids(self, ent_mask: torch.Tensor):
        max_ent = ent_mask.shape[0]
        first_appearance = -1 * torch.ones(max_ent)

        for ind in range(max_ent):
            if torch.all(ent_mask[ind] == 0):
                continue
            else:
                first_appearance[ind] = torch.where(ent_mask[ind] == 1)[0][0]

        i = first_appearance.view(1, -1)  # (1, max_ent)
        j = first_appearance.view(-1, 1)  # (max_ent, 1)

        ent_dist = j - i  # (max_ent, max_ent)

        ent_dist[(i == -1) | (j == -1)] = 0

        dist_ids = torch.tensor(np.digitize(torch.abs(ent_dist), self._dist_bins, right=False), dtype=torch.long)
        dist_ids[ent_dist < 0] *= -1

        return dist_ids

    def _extract_labels_and_mask(self, ner_facts: Tuple[EntityFact, ...], link_facts: Tuple[RelationFact, ...]):
        max_ent = self.max_ent if self.max_ent else len(ner_facts)

        fact_to_ind = {fact: ind for ind, fact in enumerate(ner_facts)}

        labels = torch.zeros(max_ent, max_ent, len(self._relations), dtype=torch.bool)
        labels[:len(ner_facts), :len(ner_facts), self._no_rel_ind] = True

        labels_mask = torch.zeros(max_ent, max_ent, dtype=torch.bool)
        labels_mask[:len(ner_facts), :len(ner_facts)] = True

        for fact in link_facts:

            if fact.fact_type_id not in self._relations or fact.from_fact not in ner_facts or fact.to_fact not in ner_facts:
                continue

            source_fact_ind = fact_to_ind[fact.from_fact]
            target_fact_ind = fact_to_ind[fact.to_fact]

            labels[source_fact_ind][target_fact_ind][self._rel_to_ind[fact.fact_type_id]] = True
            labels[source_fact_ind][target_fact_ind][self._no_rel_ind] = False

        return labels, labels_mask
