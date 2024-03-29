from typing import Any, Dict, List, Optional, TypeVar

import torch
from src.abstract import PreparedDocument
from torch.nn.utils.rnn import pad_sequence


CollatedFeatures = TypeVar('CollatedFeatures', torch.Tensor, List)


class Collator:

    @classmethod
    def collate_fn(cls, documents: List[PreparedDocument]) -> Dict[str, CollatedFeatures]:
        document_dict = {}

        for field_name in documents[0]._fields:

            if documents[0].__getattribute__(field_name) is None:
                continue

            feature_names = documents[0].__getattribute__(field_name).keys()

            for feature_name in feature_names:
                features = cls.get_features(documents, field_name, feature_name)
                document_dict[feature_name] = cls.collate(features) if features and isinstance(features[0], torch.Tensor) else features

        return document_dict

    @staticmethod
    def get_features(documents: List[PreparedDocument], field_name, feature_name) -> List[Any]:
        return [document.__getattribute__(field_name)[feature_name] for document in documents]

    @staticmethod
    def collate(tensors: Optional[List[torch.Tensor]]) -> Optional[torch.Tensor]:
        if tensors is None:
            return None

        if tensors[0].dim() == 1:  # tensors[i] has (a_i,) shape
            return pad_sequence(tensors, batch_first=True)  # (bs, a_max)

        if tensors[0].dim() == 2:  # tensors[i] has (a_i, b_i) shape

            # [t.squeeze(0) for t in tensors[i].split(1)] converts tensors[i] tensor to the list of tensors with (b_i,) shape
            # there are BS tensors; BS = \sum\limits_{i=1}^{bs} a_i
            tmp: List[torch.Tensor] = [t.squeeze(0) for ts in tensors for t in ts.split(split_size=1, dim=0)]

            tmp: torch.Tensor = pad_sequence(tmp, batch_first=True)  # (BS, b_max) shape
            tmp: List[torch.Tensor] = tmp.split([t.size()[0] for t in tensors])  # there are bs tensors with (a_i, b_max) shape
            return pad_sequence(tmp, batch_first=True)  # (bs, a_max, b_max)

        if tensors[0].dim() == 3:  # tensors[i] has (a_i, b_i, c_i) shape
            tmp: List[torch.Tensor] = [t.squeeze(0) for ts in tensors for t in ts.split(split_size=1, dim=0)]
            tmp: List[torch.Tensor] = [t.squeeze(0) for ts in tmp for t in ts.split(split_size=1, dim=0)]

            tmp: torch.Tensor = pad_sequence(tmp, batch_first=True)
            tmp: List[torch.Tensor] = tmp.split([t.size()[1] for t in tensors for _ in range(t.size()[0])])
            tmp: torch.Tensor = pad_sequence(tmp, batch_first=True)
            tmp: List[torch.Tensor] = tmp.split([t.size()[0] for t in tensors])
            return pad_sequence(tmp, batch_first=True)

        raise ValueError(f"There are too many dimensions. Expected <= 3, received {tensors[0].dim()}")
