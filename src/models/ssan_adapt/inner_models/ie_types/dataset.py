from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from src.abstract import AbstractDataset, Document, EntityFact, FactClass, NO_ENT_IND, NO_REL_IND, PreparedDocument, RelationFact, Span, \
    get_tokenizer_len_attribute

from ..base.dataset import BaseSSANAdaptDataset


class IETypesSSANAdaptDataset(BaseSSANAdaptDataset):

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

        # 1st stage: tokenize text, get input_ids and extract entity facts
        # token_to_sentence - map token_ind to the corresponding sentence_ind
        # span_to_token_ind - map span to the corresponding token_ind's
        input_ids, token_to_sentence_ind, span_to_token_ind = self._tokenize(document)
        ner_facts = self._extract_ner_facts(document, span_to_token_ind)

        if len(ner_facts) == 0:
            return

        # 2nd stage: get ner_ids and ent_mask
        # token_to_coreference_id - map token_ind to the coreference_id of the corresponding fact
        input_ids, token_to_sentence_ind, ent_mask, token_to_coreference_id = self._populate_ner_information(
            input_ids, token_to_sentence_ind, span_to_token_ind, ner_facts
        )

        # 3rd stage: get dist_ids and struct_matrix
        dist_ids = self._extract_dist_ids(ent_mask)
        struct_matrix = self._extract_struct_matrix(token_to_sentence_ind, token_to_coreference_id)

        # 4th stage: normalize ent_mask
        tmp = ent_mask.sum(dim=-1)
        tmp = tmp + (tmp == 0)
        ent_mask = ent_mask / tmp.unsqueeze(1)

        features = {
            "input_ids": input_ids,
            "dist_ids": dist_ids,
            "ent_mask": ent_mask,
            "attention_mask": torch.ones(input_ids.shape[0]).bool(),
            "struct_matrix": struct_matrix,
        }

        labels = None
        if self.extract_labels:  # 5th stage: extract labels and labels_mask
            labels_tensors, labels_mask = self._extract_labels_and_mask(ner_facts, self._extract_link_facts(document))
            labels = {"labels": labels_tensors, "labels_mask": labels_mask}

        prepared_document = PreparedDocument(features=features, labels=labels)

        self._documents.append(prepared_document)

    def _tokenize(self, document: Document):

        input_ids, token_to_sentence_ind, token_to_span = [], [], []

        for ind, sentence in enumerate(document.sentences):
            for span in sentence:
                tokens = self.tokenizer.encode(document.get_word(span))[1:-1]  # crop tokens of the beginning and the end

                tokens = tokens if len(tokens) else [self.tokenizer.unk_token_id]

                input_ids.extend(tokens)
                token_to_sentence_ind += [ind] * len(tokens)
                token_to_span += [span] * len(tokens)

        input_ids = input_ids[:self.max_len - 2]
        token_to_sentence_ind = token_to_sentence_ind[:self.max_len - 2]
        token_to_span = token_to_span[:self.max_len - 2]

        input_ids = [self._bos_token] + input_ids + [self._eos_token]
        token_to_sentence_ind = [None] + token_to_sentence_ind + [None]

        span_to_token_ind = defaultdict(list)
        for token_ind, span in enumerate(token_to_span, start=1):
            span_to_token_ind[span].append(token_ind)

        return torch.tensor(input_ids, dtype=torch.long), token_to_sentence_ind, span_to_token_ind

    def _populate_ner_information(self, input_ids: torch.Tensor, token_to_sentence_ind: List, span_to_token_ind: Dict[Span, List[int]], ner_facts: Tuple[EntityFact, ...]):
        max_ent = self.max_ent if self.max_ent else len(ner_facts)

        ent_mask = [[0] * 1 for _ in range(max_ent)]
        token_to_coreference_id = [type(self).USUAL_TOKEN] * seq_len

        for ind, fact in enumerate(ner_facts):
            ind_of_type_id = self._ent_to_ind[fact.type_id]

            for span in fact.mentions:
                for token_ind in span_to_token_ind.get(span, []):
                    ner_ids[token_ind] = ind_of_type_id
                    ent_mask[ind][token_ind] = True
                    token_to_coreference_id[token_ind] = fact.coreference_id

        ent_mask = torch.tensor(ent_mask, dtype=torch.bool)

        return input_ids, token_to_sentence_ind, ent_mask, token_to_coreference_id


    def _extract_ner_types(self, ner_facts: Tuple[EntityFact, ...], span_to_token_ind: Dict[Span, List[int]], seq_len: int):

        max_ent = self.max_ent if self.max_ent else len(ner_facts)

        ner_ids = torch.ones(seq_len, dtype=torch.long) * NO_ENT_IND
        ent_mask = torch.zeros(max_ent, seq_len, dtype=torch.bool)
        token_to_coreference_id = [type(self).USUAL_TOKEN] * seq_len

        for ind, fact in enumerate(ner_facts):
            ind_of_type_id = self._ent_to_ind[fact.type_id]

            for span in fact.mentions:
                for token_ind in span_to_token_ind.get(span, []):
                    ner_ids[token_ind] = ind_of_type_id
                    ent_mask[ind][token_ind] = True
                    token_to_coreference_id[token_ind] = fact.coreference_id

        return ner_ids, ent_mask, token_to_coreference_id
