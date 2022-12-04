from typing import Dict, List, Tuple

import torch
from src.abstract import AbstractDataset, AbstractFact, Document


class BaseSSANAdaptDataset(AbstractDataset):
    def _prepare_doc(self, doc: Document) -> \
            Tuple[Dict[str, torch.Tensor], List[AbstractFact]]:
        pass
