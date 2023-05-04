from typing import Iterable

import torch
from src.abstract import DiversifierConfig, Document

from .em import EntityMarkerDataset


class TypedEntityMarkerDataset(EntityMarkerDataset):
    def __init__(
            self,
            documents: Iterable[Document],
            tokenizer,
            extract_labels: bool,
            evaluation: bool,
            relations: Iterable[str],
            desc: str,
            diversifier: DiversifierConfig
    ):
        super(EntityMarkerDataset, self).__init__(documents, tokenizer, desc, extract_labels, evaluation, diversifier)

        self._relations = tuple(relations)
        self._rel_to_ind = {rel: ind for ind, rel in enumerate(relations)}

        self._new_tokens = []

    def _tokenize(self, document: Document):
        tokens, ss, os = [], 0, 0

        object_type = next(self._get_entity_fact(document, 'name', 'object')).type_id
        subject_type = next(self._get_entity_fact(document, 'name', 'subject')).type_id

        if self.diversifier.active:
            object_type = self.diversifier[object_type]
            subject_type = self.diversifier[subject_type]

        object_end_token = '[/OBJ-{}]'.format(object_type)
        subject_end_token = '[/SUBJ-{}]'.format(subject_type)
        object_start_token = '[OBJ-{}]'.format(object_type)
        subject_start_token = '[SUBJ-{}]'.format(subject_type)

        for token in (object_end_token, subject_end_token, object_start_token, subject_start_token):
            if token not in self._new_tokens:
                self._new_tokens.append(token)
                self.tokenizer.add_tokens([token])

        subject_start, subject_end, object_start, object_end = self._get_facts_info(document)

        for word in document.words:
            word_tokens = self.word2token(word.text)

            if word == subject_start:
                ss = len(tokens)
                word_tokens = [subject_start_token] + word_tokens
            elif word == subject_end:
                word_tokens = word_tokens + [subject_end_token]
            elif word == object_start:
                os = len(tokens)
                word_tokens = [object_start_token] + word_tokens
            elif word == object_end:
                word_tokens = word_tokens + [object_end_token]

            tokens.extend(word_tokens)

        tokens = tokens[:self.max_len - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor([ss + 1], dtype=torch.long), torch.tensor([os + 1], dtype=torch.long)
