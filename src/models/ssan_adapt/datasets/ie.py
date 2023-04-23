import itertools
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from src.abstract import Document, EntityFact, FactClass, NO_ENT_IND, PreparedDocument, Span

from .base import BaseDataset


class IETypesDataset(BaseDataset):
    ENT_IND = NO_ENT_IND + 1

    def __getitem__(self, idx: int) -> PreparedDocument:
        if len(self._prepared_docs) > 0:  # if there is already prepared doc in memory then returns it (dev and test datasets)
            return super(IETypesDataset, self).__getitem__(idx)
        return self._prepare_document(self._documents[idx])

    def _get_ent_tokens(self, document: Document):

        spans = tuple(itertools.chain.from_iterable(document.sentences))
        spans_to_ind = {span: ind for ind, span in enumerate(spans)}

        start, end = [()] * len(spans), [()] * len(spans)

        ner_facts: Tuple[EntityFact, ...] = tuple(filter(lambda fact: fact.fact_class is FactClass.ENTITY, document.facts))[:self.max_ent]

        for ner_fact in ner_facts:

            type_id = ner_fact.type_id if not self.diversifier.active else self.diversifier[ner_fact.type_id]
            fact_info = (ner_fact.coreference_id, type_id)

            for span in ner_fact.mentions:
                span_ind = spans_to_ind[span]

                if span_ind > 0:
                    if end[span_ind - 1] != fact_info:
                        start[span_ind] = fact_info
                    else:
                        end[span_ind - 1] = ()
                else:
                    start[span_ind] = fact_info

                if span_ind < len(spans) - 1:
                    if start[span_ind + 1] != fact_info:
                        end[span_ind] = fact_info
                    else:
                        start[span_ind + 1] = ()
                else:
                    end[span_ind] = fact_info

        start_ent_tokens = [f' <{s[1]}> ' if s else '' for s in start]
        end_ent_tokens = [f' </{e[1]}> ' if e else '' for e in end]

        return start_ent_tokens, end_ent_tokens

    def _tokenize(self, document: Document):

        input_ids, token_to_sentence_ind, token_to_span = [], [], []
        start_ent_tokens, end_ent_tokens = self._get_ent_tokens(document)

        cur_ind = 0
        for ind, sentence in enumerate(document.sentences):
            for span in sentence:
                word = start_ent_tokens[cur_ind] + document.get_word(span) + end_ent_tokens[cur_ind]
                tokens = self._word2token(word)

                input_ids.extend(tokens)
                token_to_sentence_ind += [ind] * len(tokens)
                token_to_span += [span] * len(tokens)
                cur_ind += 1

        input_ids = input_ids[:self.max_len - 2]
        token_to_sentence_ind = token_to_sentence_ind[:self.max_len - 2]
        token_to_span = token_to_span[:self.max_len - 2]

        input_ids = [self._bos_token] + input_ids + [self._eos_token]
        token_to_sentence_ind = [-1] + token_to_sentence_ind + [-1]

        span_to_token_ind = defaultdict(list)
        for token_ind, span in enumerate(token_to_span, start=1):
            span_to_token_ind[span].append(token_ind)

        return torch.tensor(input_ids, dtype=torch.long), token_to_sentence_ind, span_to_token_ind

    def _extract_ner_types(self, ner_facts: Tuple[EntityFact, ...], span_to_token_ind: Dict[Span, List[int]], seq_len: int):

        max_ent = self.max_ent if self.max_ent else len(ner_facts)

        ner_ids = torch.ones(seq_len, dtype=torch.long) * NO_ENT_IND
        ent_mask = torch.zeros(max_ent, seq_len, dtype=torch.bool)
        token_to_coreference_id = [type(self).USUAL_TOKEN] * seq_len

        ind_of_type_id = type(self).ENT_IND
        for ind, fact in enumerate(ner_facts):
            for span in fact.mentions:
                for token_ind in span_to_token_ind.get(span, []):
                    ner_ids[token_ind] = ind_of_type_id
                    ent_mask[ind][token_ind] = True
                    token_to_coreference_id[token_ind] = fact.coreference_id

        return ner_ids, ent_mask, token_to_coreference_id
