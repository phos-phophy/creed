from abc import ABCMeta, abstractmethod
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .diversifier import Diversifier, DiversifierConfig
from .document import Document
from .helpers import get_tokenizer_len_attribute


class PreparedDocument(NamedTuple):
    features: Dict[str, torch.Tensor]
    labels: Optional[Dict[str, torch.Tensor]]  # e.g. {"labels": ..., "labels_mask": ...}


class AbstractDataset(Dataset, metaclass=ABCMeta):
    def __init__(
            self,
            documents: Iterable[Document],
            tokenizer,
            desc: str,
            extract_labels: bool,
            evaluation: bool,
            diversifier: DiversifierConfig
    ):
        super(AbstractDataset, self).__init__()

        self._tokenizer = tokenizer

        self._desc = desc
        self._extract_labels = extract_labels
        self._evaluation = evaluation
        self._evaluation = evaluation

        self._epoch = None

        self._diversifier = Diversifier(tokenizer, diversifier)

        self._len_attr = get_tokenizer_len_attribute(tokenizer)

        self._documents: Tuple[Document] = tuple(documents)
        self._prepared_docs: List[PreparedDocument] = []
        self._used_docs = 0

    def __getitem__(self, idx: int) -> PreparedDocument:
        self.prepare_documents()
        self._used_docs += 1

        return self._prepared_docs[idx]

    def __len__(self):
        return len(self._documents)

    @property
    def diversifier(self):
        return self._diversifier

    @property
    def epoch(self):
        return self._epoch

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
    def should_prepare(self):
        return self.diversifier.active and self._used_docs >= len(self._documents) or len(self._prepared_docs) == 0

    @property
    def tokenizer(self):
        return self._tokenizer

    @abstractmethod
    def _prepare_document(self, document: Document) -> PreparedDocument:
        pass

    def prepare_documents(self):
        if self.should_prepare:
            documents = tqdm(self._documents, desc=self._desc, disable=not bool(self._desc))
            self._prepared_docs = list(map(self._prepare_document, documents))
            self._used_docs = 0
        return self
