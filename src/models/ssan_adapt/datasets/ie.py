from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from src.abstract import Document, EntityFact, NO_ENT_IND, PreparedDocument, Word

from .base import BaseDataset


class IETypesDataset(BaseDataset):
    ENT_IND = NO_ENT_IND + 1

    def __getitem__(self, idx: int) -> PreparedDocument:
        if len(self._prepared_docs) > 0:  # if there is already prepared doc in memory then returns it (dev and test datasets)
            return super(IETypesDataset, self).__getitem__(idx)
        return self._prepare_document(self._documents[idx])

    def _get_ent_tokens(self, document: Document):

        words = document.words
        start, end = [()] * len(words), [()] * len(words)

        for ner_fact in document.entity_facts[:self.max_ent]:

            type_id = ner_fact.type_id if not self.diversifier.active else self.diversifier[ner_fact.type_id]
            fact_info = (ner_fact.coreference_id, type_id)

            for mention in ner_fact.mentions:
                for word in mention.words:
                    if word.ind_in_doc > 0:
                        if end[word.ind_in_doc - 1] != fact_info:
                            start[word.ind_in_doc] = fact_info
                        else:
                            end[word.ind_in_doc - 1] = ()
                    else:
                        start[word.ind_in_doc] = fact_info

                    if word.ind_in_doc < len(words) - 1:
                        if start[word.ind_in_doc + 1] != fact_info:
                            end[word.ind_in_doc] = fact_info
                        else:
                            start[word.ind_in_doc + 1] = ()
                    else:
                        end[word.ind_in_doc] = fact_info

        start_ent_tokens = [f' <{s[1]}> ' if s else '' for s in start]
        end_ent_tokens = [f' </{e[1]}> ' if e else '' for e in end]

        return start_ent_tokens, end_ent_tokens

    def _tokenize(self, document: Document):

        tokens, token_to_sentence_ind, token_to_word = [], [], []
        start_ent_tokens, end_ent_tokens = self._get_ent_tokens(document)

        for word in document.words:
            s_tokens = self.word2token(start_ent_tokens[word.ind_in_doc]) if len(start_ent_tokens[word.ind_in_doc]) else []
            e_tokens = self.word2token(end_ent_tokens[word.ind_in_doc]) if len(end_ent_tokens[word.ind_in_doc]) else []

            word_tokens = s_tokens + self.word2token(word.text) + e_tokens

            tokens.extend(word_tokens)
            token_to_sentence_ind += [word.sent_ind] * len(word_tokens)
            token_to_word += [word] * len(word_tokens)

        tokens = tokens[:self.max_len - 2]
        token_to_sentence_ind = token_to_sentence_ind[:self.max_len - 2]
        token_to_word = token_to_word[:self.max_len - 2]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        token_to_sentence_ind = [-1] + token_to_sentence_ind + [-1]

        word_to_token_ind = defaultdict(list)
        for token_ind, word in enumerate(token_to_word, start=1):
            word_to_token_ind[word].append(token_ind)

        return torch.tensor(input_ids, dtype=torch.long), token_to_sentence_ind, word_to_token_ind

    def _extract_ner_types(self, ner_facts: Tuple[EntityFact, ...], word_to_token_ind: Dict[Word, List[int]], seq_len: int):

        max_ent = self.max_ent if self.max_ent else len(ner_facts)

        ner_ids = torch.ones(seq_len, dtype=torch.long) * NO_ENT_IND
        ent_mask = torch.zeros(max_ent, seq_len, dtype=torch.bool)
        token_to_coreference_id = [type(self).USUAL_TOKEN] * seq_len

        ind_of_type_id = type(self).ENT_IND
        for ind, fact in enumerate(ner_facts):
            for mention in fact.mentions:
                for word in mention.words:
                    for token_ind in word_to_token_ind.get(word, []):
                        ner_ids[token_ind] = ind_of_type_id
                        ent_mask[ind][token_ind] = True
                        token_to_coreference_id[token_ind] = fact.coreference_id

        return ner_ids, ent_mask, token_to_coreference_id
