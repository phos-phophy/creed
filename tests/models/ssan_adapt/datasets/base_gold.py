from typing import List

import torch
from src.models.ssan_adapt.datasets import BaseDataset


class GoldDataset(BaseDataset):
    def _extract_struct_matrix(self, token_to_sentence_ind: List[int], token_to_coreference_id: List[int]):
        length = len(token_to_sentence_ind)
        struct_mask = torch.zeros((5, length, length), dtype=torch.bool)

        for i in range(length):

            if token_to_coreference_id[i] == type(self).USUAL_TOKEN:
                continue

            for j in range(length):

                if token_to_sentence_ind[i] != token_to_sentence_ind[j]:
                    if token_to_coreference_id[i] == token_to_coreference_id[j]:
                        struct_mask[0][i][j] = True  # inter-coref
                    elif token_to_coreference_id[j] != type(self).USUAL_TOKEN:
                        struct_mask[1][i][j] = True  # inter-relate
                else:
                    if token_to_coreference_id[i] == token_to_coreference_id[j]:
                        struct_mask[2][i][j] = True  # intra-coref
                    elif token_to_coreference_id[j] != type(self).USUAL_TOKEN:
                        struct_mask[3][i][j] = True  # intra-relate
                    else:
                        struct_mask[4][i][j] = True  # intra-NA

        return struct_mask
