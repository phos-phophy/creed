from collections import defaultdict
from typing import Dict, List, Optional

import torch
from src.abstract.examples import PreparedDocument
from torch.nn.utils.rnn import pad_sequence


def collate_fn(items: List[PreparedDocument]) -> PreparedDocument:
    document_dict = defaultdict(dict)

    for field in items[0]._fields:

        keys = items[0].__getattribute__(field).keys()

        features: Dict[str, List[torch.Tensor]] = {key: [element[key] for element in items] for key in keys}
        for key, feature in features.items():
            document_dict[field][key] = collate(feature)

    return PreparedDocument(**document_dict)


def collate(tensors: Optional[List[torch.Tensor]]):
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
