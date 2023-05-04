from itertools import product
from typing import Dict, Iterable, List, Tuple

import torch
from src.abstract import AbstractDataset, DiversifierConfig, Document, EntityFact, NO_REL_IND, PreparedDocument, Word


class BaseDataset(AbstractDataset):

    def __init__(
            self,
            documents: Iterable[Document],
            tokenizer,
            desc: str,
            extract_labels: bool,
            evaluation: bool,
            diversifier: DiversifierConfig,
            relations: Iterable[str]
    ):
        super(BaseDataset, self).__init__(documents, tokenizer, desc, extract_labels, evaluation, diversifier)

        self._entity_type = {'ORG': 1, 'LOC': 3, 'TIME': 5, 'PER': 7, 'MISC': 9, 'NUM': 11}

        self._relations = tuple(relations)
        self._rel_to_ind = {rel: ind for ind, rel in enumerate(relations)}

    def _prepare_document(self, document: Document) -> PreparedDocument:
        """
        1) input_ids: LongTensor (len,)
        2) attention_mask: BoolTensor (len,)
        3) entity_pos: List[List[Tuple[int, int]]]
        4) hts: LongTensor (ent * (ent - 1), max_ent)
        5) labels: BoolTensor (ent * (ent - 1), num_link)
        """

        input_ids, sent_map = self._tokenize(document, document.entity_facts)
        entity_pos = self._get_entity_pos(document.entity_facts, sent_map)
        labels, hts = self._get_labels_and_hts(document, document.entity_facts)

        features = {
            'input_ids': input_ids,
            'attention_mask': torch.ones(input_ids.shape[0], dtype=torch.bool),
            'entity_pos': entity_pos,
            'hts': hts,
        }

        return PreparedDocument(features=features, labels={"labels": labels})

    def _get_ent_tokens(self, document: Document, ner_facts: Tuple[EntityFact, ...]) -> Tuple[List[str], List[str]]:

        words = document.words
        start, end = [()] * len(words), [()] * len(words)

        for ner_fact in ner_facts:

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

        start_ent_tokens = [f' [unused{self._entity_type[s[1]]}] ' if s else '' for s in start]
        end_ent_tokens = [f' [unused{self._entity_type[e[1]] + 50}] ' if e else '' for e in end]

        return start_ent_tokens, end_ent_tokens

    def _tokenize(self, document: Document, ner_facts: Tuple[EntityFact, ...]) -> Tuple[torch.Tensor, Dict[Word, int]]:
        tokens, word_map = [], {}
        start_ent_tokens, end_ent_tokens = self._get_ent_tokens(document, ner_facts)

        for word in document.words:
            s_tokens = self.word2token(start_ent_tokens[word.ind_in_doc]) if len(start_ent_tokens[word.ind_in_doc]) else []
            e_tokens = self.word2token(end_ent_tokens[word.ind_in_doc]) if len(end_ent_tokens[word.ind_in_doc]) else []

            word_tokens = s_tokens + self.word2token(word.text) + e_tokens

            word_map[word] = len(tokens)
            tokens.extend(word_tokens)

        tokens = tokens[:self.max_len - 2]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)

        return torch.tensor(input_ids, dtype=torch.long), word_map

    @staticmethod
    def _get_entity_pos(ner_facts: Tuple[EntityFact, ...], word_map: Dict[Word, int]) -> List[List[Tuple[int, int]]]:
        entity_pos = []

        for fact in ner_facts:
            entity_pos.append([])

            for mention in fact.mentions:
                start = word_map[mention.words[0].ind_in_doc]
                end = word_map[mention.words[-1].ind_in_doc]
                entity_pos[-1].append((start, end))

        return entity_pos

    def _get_labels_and_hts(self, document: Document, ner_facts: Tuple[EntityFact, ...]) -> Tuple[torch.BoolTensor, torch.LongTensor]:

        n = len(ner_facts)
        fact_to_ind = {fact: ind for ind, fact in enumerate(ner_facts)}

        hts = torch.tensor(tuple(product(range(n), repeat=2)), dtype=torch.long)
        labels = torch.zeros(n * n, len(self._relations), dtype=torch.bool)
        labels[:, NO_REL_IND] = True

        for rel_fact in document.relation_facts:
            source_fact_ind = fact_to_ind[rel_fact.from_fact]
            target_fact_ind = fact_to_ind[rel_fact.to_fact]

            ind = source_fact_ind * n + target_fact_ind

            labels[ind][self._rel_to_ind[rel_fact.type_id]] = True
            labels[ind][NO_REL_IND] = False

        mask = hts[:, 0] != hts[:, 1]
        hts = hts[mask]
        labels = labels[mask]

        return labels, hts
