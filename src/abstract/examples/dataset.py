from abc import ABCMeta, abstractmethod
from typing import Dict, Iterable, List, Tuple

import torch
from src.abstract.features.fact import AbstractFact
from torch.utils.data import Dataset

from .document import Document


class AbstractDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, docs: Iterable[Document], tokenizer, extract_labels: bool, evaluation: bool):
        super(AbstractDataset, self).__init__()

        self._tokenizer = tokenizer
        self._extract_labels = extract_labels
        self._evaluation = evaluation

        self._facts: List[List[AbstractFact]] = []
        self._prepared_docs: List[Dict[str, torch.Tensor]] = []

        for doc in docs:
            prepared_doc, extracted_facts = self._prepare_doc(doc)
            self._prepared_docs.append(prepared_doc)
            self._facts.append(extracted_facts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._prepared_docs[idx]

    def __len__(self):
        return len(self._prepared_docs)

    @property
    def extract_labels(self):
        return self._extract_labels

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def evaluation(self):
        return self._evaluation

    def get_fact(self, doc_idx: int, fact_idx: int) -> AbstractFact:
        return self._facts[doc_idx][fact_idx]

    def fact_count(self, doc_idx: int):
        return len(self._facts[doc_idx])

    @abstractmethod
    def _prepare_doc(self, doc: Document) -> \
            Tuple[Dict[str, torch.Tensor], List[AbstractFact]]:
        pass
