from typing import List, Tuple

from src.abstract import Document, EntityFact

from .base import BaseDataset


class IETypesDataset(BaseDataset):
    def _get_ent_tokens(self, document: Document, ner_facts: Tuple[EntityFact, ...]) -> Tuple[List[str], List[str]]:

        words = document.words
        start, end = [()] * len(words), [()] * len(words)

        for ner_fact in ner_facts:

            type_id = ner_fact.type_id if not self.diversifier.active else self.diversifier[ner_fact.type_id]
            fact_info = (ner_fact.coreference_id, type_id)

            for mention in ner_fact.mentions:
                for word in mention.words:
                    if word.ind_in_doc > 0:
                        if end[word.ind_in_doc - 1] != fact_info:
                            start[word.ind_in_doc] = fact_info
                        else:
                            end[word.ind_in_doc - 1] = ()
                    else:
                        start[word.ind_in_doc] = fact_info

                    if word.ind_in_doc < len(words) - 1:
                        if start[word.ind_in_doc + 1] != fact_info:
                            end[word.ind_in_doc] = fact_info
                        else:
                            start[word.ind_in_doc + 1] = ()
                    else:
                        end[word.ind_in_doc] = fact_info

        start_ent_tokens = [f' [{s[1]}] ' if s else '' for s in start]
        end_ent_tokens = [f' [/{e[1]}] ' if e else '' for e in end]

        return start_ent_tokens, end_ent_tokens
