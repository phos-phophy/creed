from typing import Dict, Tuple

import torch
from src.abstract import Document, EntityFact, Word

from .base import BaseDataset


class WOTypesDataset(BaseDataset):
    def _tokenize(self, document: Document, ner_facts: Tuple[EntityFact, ...]) -> Tuple[torch.Tensor, Dict[Word, int]]:
        tokens, word_map = [], {}

        for word in document.words:
            word_tokens = self.word2token(word.text)

            word_map[word] = len(tokens)
            tokens.extend(word_tokens)

        tokens = tokens[:self.max_len - 2]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)

        return torch.tensor(input_ids, dtype=torch.long), word_map
