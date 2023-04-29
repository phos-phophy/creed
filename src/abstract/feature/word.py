from functools import total_ordering
from typing import Iterable


@total_ordering
class Word:
    def __init__(self, text: str, sent_ind: int, ind_in_sent: int, ind_in_doc: int):
        self._text = text
        self._sent_ind = sent_ind
        self._ind_in_sent = ind_in_sent
        self._ind_in_doc = ind_in_doc

    def __eq__(self, other: 'Word'):
        return self.text == other.text and self.ind_in_doc == other.ind_in_doc

    def __lt__(self, other: 'Word'):
        return self.ind_in_doc < other.ind_in_doc

    def __hash__(self):
        return hash((self.text, self.sent_ind, self.ind_in_sent, self.ind_in_doc))

    def __len__(self):
        return len(self.text)

    def __repr__(self):
        return self.text

    @property
    def text(self):
        return self._text

    @property
    def sent_ind(self):
        return self._sent_ind

    @property
    def ind_in_sent(self):
        return self._ind_in_sent

    @property
    def ind_in_doc(self):
        return self._ind_in_doc


class Mention:
    def __init__(self, words: Iterable[Word]):
        self._words = tuple(sorted(words))

    @property
    def words(self):
        return self._words

    def __eq__(self, other: 'Mention'):
        return self.words == other.words

    def __hash__(self):
        return hash(self.words)

    def __repr__(self):
        return 'Mention(' + ', '.join(str(word) for word in self.words) + ')'
