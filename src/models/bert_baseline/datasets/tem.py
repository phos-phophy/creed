import torch
from src.abstract import Document

from .em import EntityMarkerDataset


class TypedEntityMarkerDataset(EntityMarkerDataset):

    def _tokenize(self, document: Document):
        input_ids, ss, os = [], 0, 0

        object_type = next(self._get_fact(document, 'name', 'object')).type_id
        subject_type = next(self._get_fact(document, 'name', 'subject')).type_id

        if self.diversifier.active:
            object_type = self.diversifier[object_type]
            subject_type = self.diversifier[subject_type]

        subject_start_token, subject_end_token, object_start_token, object_end_token = self._get_facts_info(document)

        for sentence in document.sentences:
            for span in sentence:

                if span == subject_start_token:
                    ss = len(input_ids)
                    tokens = self._word2token('[SUBJ-{}] '.format(subject_type) + document.get_word(span))
                elif span == subject_end_token:
                    tokens = self._word2token(document.get_word(span) + ' [/SUBJ-{}]'.format(subject_type))
                elif span == object_start_token:
                    os = len(input_ids)
                    tokens = self._word2token('[OBJ-{}] '.format(object_type) + document.get_word(span))
                elif span == subject_end_token:
                    tokens = self._word2token(document.get_word(span) + ' [/OBJ-{}]'.format(object_type))
                else:
                    tokens = self._word2token(document.get_word(span))

                input_ids.extend(tokens)

        input_ids = [self._bos_token] + input_ids[:self.max_len - 2] + [self._eos_token]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor([ss + 1], dtype=torch.long), torch.tensor([os + 1], dtype=torch.long)
