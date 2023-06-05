from typing import Any, Iterable

import torch
from src.abstract import AbstractDataset, DiversifierConfig, Document, EntityFact, Mention, NO_REL_IND, PreparedDocument


class EntityMarkerDataset(AbstractDataset):
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

        self.tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]'])

        self._relations = tuple(relations)
        self._rel_to_ind = {rel: ind for ind, rel in enumerate(relations)}

    def _prepare_document(self, document: Document) -> PreparedDocument:

        input_ids, ss, os = self._tokenize(document)

        features = {
            "input_ids": input_ids,
            "attention_mask": torch.ones(input_ids.shape[0]).bool(),
            "ss": ss,
            "os": os
        }

        labels = None
        if self.extract_labels:
            link_fact = document.relation_facts[0] if len(document.relation_facts) else None
            rel_ind = self._rel_to_ind[link_fact.type_id] if link_fact else NO_REL_IND
            labels = {"labels": torch.tensor([rel_ind], dtype=torch.long)}

        return PreparedDocument(features=features, labels=labels)

    @staticmethod
    def _get_entity_fact(document: Document, attribute_name: str, attribute_value: Any):
        for fact in document.entity_facts:
            if hasattr(fact, attribute_name) and fact.__getattribute__(attribute_name) == attribute_value:
                yield fact
        yield None

    def _get_facts_info(self, document: Document):
        object_fact: EntityFact = next(self._get_entity_fact(document, 'name', 'object'))
        subject_fact: EntityFact = next(self._get_entity_fact(document, 'name', 'subject'))

        object_mention: Mention = next(iter(object_fact.mentions))
        subject_mention: Mention = next(iter(subject_fact.mentions))

        return subject_mention.words[0], subject_mention.words[-1], object_mention.words[0], object_mention.words[-1]

    def _tokenize(self, document: Document):
        tokens, ss, os = [], 0, 0

        subject_start, subject_end, object_start, object_end = self._get_facts_info(document)

        for word in document.words:
            word_tokens = self.word2token(word.text)

            if word == subject_start:
                tmp = len(tokens) + 1
                ss = [tmp, tmp + 1]
                word_tokens = ['[E1]'] + word_tokens
            elif word == subject_end:
                word_tokens = word_tokens + ['[/E1]']
            elif word == object_start:
                tmp = len(tokens) + 1
                os = [tmp, tmp + 1]
                word_tokens = ['[E2]'] + word_tokens
            elif word == object_end:
                word_tokens = word_tokens + ['[/E2]']

            tokens.extend(word_tokens)

        tokens = tokens[:self.max_len - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(ss, dtype=torch.long), torch.tensor(os, dtype=torch.long)
