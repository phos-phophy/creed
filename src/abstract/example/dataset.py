import pickle as pkl
from abc import ABCMeta, abstractmethod
from pathlib import Path
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
            diversifier: DiversifierConfig,
            cache_dir: Path = None,
            dataset_name: str = ''
    ):
        super(AbstractDataset, self).__init__()

        self._tokenizer = tokenizer

        self._desc = desc
        self._extract_labels = extract_labels
        self._evaluation = evaluation
        self._cache_dir = cache_dir
        self._dataset_name = dataset_name

        self._epoch = None

        self._diversifier = Diversifier(tokenizer, diversifier)

        self._len_attr = get_tokenizer_len_attribute(tokenizer)

        self._documents: Tuple[Document] = tuple(documents)
        self._prepared_docs: List[PreparedDocument] = []
        self._used_docs = 0

    def __getitem__(self, idx: int) -> PreparedDocument:
        if self.should_prepare:
            self.prepare_documents()

        self._used_docs += 1

        return self._prepared_docs[idx]

    def __len__(self):
        return len(self._documents)

    @property
    def cache_dir(self):
        return self._cache_dir

    @property
    def dataset_name(self):
        return self._dataset_name

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

    @property
    def cache_file(self):
        if self.cache_dir:
            return self.cache_dir / str(self.__class__.__name__) / (self.dataset_name.replace('/', '_').replace('.', '_') + '.pkl')
        return ''

    @abstractmethod
    def _prepare_document(self, document: Document) -> PreparedDocument:
        pass

    def prepare_documents(self):
        if self._load_documents():
            return self
        documents = tqdm(self._documents, desc=self._desc, disable=not bool(self._desc))
        self._prepared_docs = list(map(self._prepare_document, documents))
        self._used_docs = 0
        self._save_documents()
        return self

    def _load_documents(self) -> bool:
        if self.diversifier.active or not self.cache_dir:
            return False

        cache_file = self.cache_file
        if not cache_file.exists():
            return False

        with cache_file.open('rb') as f:
            documents: List[PreparedDocument] = pkl.load(f)

        if len(documents) == len(self._documents):
            print(f'Loaded cached dataset from {cache_file}')
            self._prepared_docs = documents
            self._used_docs = 0
            return True

        return False

    def _save_documents(self):
        if self.diversifier.active or not self.cache_dir:
            return

        cache_file = self.cache_file
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        with cache_file.open('wb') as file:
            pkl.dump(self._prepared_docs, file)
