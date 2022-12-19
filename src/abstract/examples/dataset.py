from abc import ABCMeta, abstractmethod
from typing import Dict, Iterable, List, NamedTuple, Optional

import torch
from torch.utils.data import Dataset

from .document import Document


class PreparedDocument(NamedTuple):
    features: Dict[str, torch.Tensor]
    labels: Optional[Dict[str, torch.Tensor]]  # e.g. {"labels": ..., "labels_mask": ...}


class AbstractDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, documents: Iterable[Document], tokenizer, extract_labels: bool, evaluation: bool):
        super(AbstractDataset, self).__init__()

        self._tokenizer = tokenizer
        self._extract_labels = extract_labels
        self._evaluation = evaluation

        self._setup_len_attr(tokenizer)

        self._documents: List[PreparedDocument] = []
        for doc in documents:
            self._prepare_document(doc)

    def __getitem__(self, idx: int) -> PreparedDocument:
        return self._documents[idx]

    def __len__(self):
        return len(self._documents)

    @property
    def evaluation(self):
        return self._evaluation

    @property
    def extract_labels(self):
        return self._extract_labels

    @property
    def max_len(self):
        return self.tokenizer.__getattribute__(self._len_attr)

    @property
    def tokenizer(self):
        return self._tokenizer

    def _setup_len_attr(self, tokenizer):
        self._len_attr = None
        for attr in tokenizer.__dict__:
            if 'max_len' in attr:
                self._len_attr = attr
                break

        if self._len_attr is None:
            raise ValueError("Can not find max_length attribute in tokenizer object")

    @abstractmethod
    def _prepare_document(self, document: Document):
        pass
