from pathlib import Path
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
            diversifier: DiversifierConfig,
            cache_dir: Path = None,
            dataset_name: str = ''
    ):
        super(EntityMarkerDataset, self).__init__(
            documents, tokenizer, desc, extract_labels, evaluation, diversifier, cache_dir, dataset_name
        )

        self._relations = tuple(relations)
        self._rel_to_ind = {rel: ind for ind, rel in enumerate(relations)}

        self._bos_token = self.tokenizer.cls_token_id
        self._eos_token = self.tokenizer.sep_token_id

        self._local_cache = dict()

        self._new_tokens = []

    def _tokenize(self, document: Document):
        input_ids, ss, os = [], 0, 0

        object_type = next(self._get_fact(document, 'name', 'object')).type_id
        subject_type = next(self._get_fact(document, 'name', 'subject')).type_id

        if self.diversifier.active:
            object_type = self.diversifier[object_type]
            subject_type = self.diversifier[subject_type]

        object_end = '[/OBJ-{}]'.format(object_type)
        subject_end = '[/SUBJ-{}]'.format(subject_type)
        object_start = '[OBJ-{}]'.format(object_type)
        subject_start = '[SUBJ-{}]'.format(subject_type)

        for token in (object_end, subject_end, object_start, subject_start):
            if token not in self._new_tokens:
                self._new_tokens.append(token)
                self.tokenizer.add_tokens([token])

        subject_start_token, subject_end_token, object_start_token, object_end_token = self._get_facts_info(document)

        for sentence in document.sentences:
            for span in sentence:

                tokens = self._word2token(document.get_word(span))

                if span == subject_start_token:
                    ss = len(input_ids)
                    tokens = subject_start + tokens
                elif span == subject_end_token:
                    tokens = tokens + subject_end
                elif span == object_start_token:
                    os = len(input_ids)
                    tokens = object_start + tokens
                elif span == object_end_token:
                    tokens = tokens + object_end

                input_ids.extend(tokens)

        input_ids = [self._bos_token] + input_ids[:self.max_len - 2] + [self._eos_token]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor([ss + 1], dtype=torch.long), torch.tensor([os + 1], dtype=torch.long)
