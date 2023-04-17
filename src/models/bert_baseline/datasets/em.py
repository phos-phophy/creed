from pathlib import Path
from typing import Any, Iterable

import torch
from src.abstract import AbstractDataset, DiversifierConfig, Document, EntityFact, FactClass, NO_REL_IND, PreparedDocument


class EntityMarkerDataset(AbstractDataset):
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

    def _prepare_document(self, document: Document) -> PreparedDocument:

        input_ids, ss, os = self._tokenize(document)

        features = {
            "input_ids": input_ids,
            "ner_ids": torch.zeros(input_ids.shape[0]),
            "attention_mask": torch.ones(input_ids.shape[0]).bool(),
            "ss": ss,
            "os": os
        }

        labels = None
        if self.extract_labels:
            link_fact = next(self._get_fact(document, 'fact_class', FactClass.RELATION))
            labels = {"labels": self._rel_to_ind[link_fact.type_id] if link_fact else NO_REL_IND}

        return PreparedDocument(features=features, labels=labels)

    def _word2token(self, word_: str):
        if word_ not in self._local_cache:
            tokens_ = self.tokenizer.encode(word_)[1:-1]  # crop tokens of the beginning and the end
            self._local_cache[word_] = tokens_ if len(tokens_) else [self.tokenizer.unk_token_id]
        return self._local_cache[word_]

    @staticmethod
    def _get_fact(document: Document, attribute_name: str, attribute_value: Any):
        for fact in document.facts:
            if hasattr(fact, attribute_name) and fact.__getattribute__(attribute_name) == attribute_value:
                yield fact
        yield None

    def _get_facts_info(self, document: Document):
        object_fact: EntityFact = next(self._get_fact(document, 'name', 'object'))
        subject_fact: EntityFact = next(self._get_fact(document, 'name', 'subject'))

        object_spans = sorted(object_fact.mentions)
        subject_spans = sorted(subject_fact.mentions)

        return subject_spans[0], subject_spans[-1], object_spans[0], object_spans[-1]

    def _tokenize(self, document: Document):
        input_ids, ss, os = [], 0, 0

        subject_start_token, subject_end_token, object_start_token, object_end_token = self._get_facts_info(document)

        for sentence in document.sentences:
            for span in sentence:

                if span == subject_start_token:
                    ss = len(input_ids)
                    tokens = self._word2token('[E1] ' + document.get_word(span))
                elif span == subject_end_token:
                    tokens = self._word2token(document.get_word(span) + ' [/E1]')
                elif span == object_start_token:
                    os = len(input_ids)
                    tokens = self._word2token('[E2] ' + document.get_word(span))
                elif span == subject_end_token:
                    tokens = self._word2token(document.get_word(span) + ' [/E2]')
                else:
                    tokens = self._word2token(document.get_word(span))

                input_ids.extend(tokens)

        input_ids = [self._bos_token] + input_ids[:self.max_len - 2] + [self._eos_token]

        return torch.tensor(input_ids, dtype=torch.long), ss + 1, os + 1
