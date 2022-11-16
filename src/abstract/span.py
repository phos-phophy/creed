class Span:
    def __init__(self, text: str, start_idx: int, end_idx: int):
        self._text = text
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._validate_length()

    def __contains__(self, item: 'Span'):
        start = item.start_idx - self.start_idx
        return self.start_idx <= item.start_idx and self.end_idx >= item.end_idx and item.text == self.text[start: start + len(item)]

    def __eq__(self, other: 'Span'):
        return self.text == other.text and self.start_idx == other.start_idx and self.end_idx == self.end_idx

    def __len__(self):
        return len(self.text)

    def __repr__(self):
        return f"({self.start_idx}, {self.end_idx})"

    @property
    def text(self):
        return self._text

    @property
    def start_idx(self):
        return self._start_idx

    @property
    def end_idx(self):
        return self._end_idx

    def _validate_length(self):
        if len(self) != (self.end_idx - self.start_idx):
            raise ValueError(f"Span length does not equal length of the span's text: {len(self)} != {self.end_idx - self.start_idx}")
