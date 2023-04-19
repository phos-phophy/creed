from typing import Dict, List, Tuple

import torch
from src.abstract import EntityFact, NO_ENT_IND, Span

from .base import BaseDataset


class WOTypesDataset(BaseDataset):
    ENT_IND = NO_ENT_IND + 1

    def _extract_ner_types(self, ner_facts: Tuple[EntityFact, ...], span_to_token_ind: Dict[Span, List[int]], seq_len: int):

        max_ent = self.max_ent if self.max_ent else len(ner_facts)

        ner_ids = torch.ones(seq_len, dtype=torch.long) * NO_ENT_IND
        ent_mask = torch.zeros(max_ent, seq_len, dtype=torch.bool)
        token_to_coreference_id = [type(self).USUAL_TOKEN] * seq_len

        ind_of_type_id = type(self).ENT_IND
        for ind, fact in enumerate(ner_facts):
            for span in fact.mentions:
                for token_ind in span_to_token_ind.get(span, []):
                    ner_ids[token_ind] = ind_of_type_id
                    ent_mask[ind][token_ind] = True
                    token_to_coreference_id[token_ind] = fact.coreference_id

        return ner_ids, ent_mask, token_to_coreference_id
