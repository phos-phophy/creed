class Span:
    def __init__(self, start_idx: int, end_idx: int):
        self._start_idx = start_idx
        self._end_idx = end_idx

    def __contains__(self, item: 'Span'):
        return self.start_idx <= item.start_idx and self.end_idx >= item.end_idx

    def __eq__(self, other: 'Span'):
        return self.start_idx == other.start_idx and self.end_idx == self.end_idx

    def __hash__(self):
        return hash((self.start_idx, self.end_idx))

    def __len__(self):
        return self.end_idx - self.end_idx

    def __repr__(self):
        return f"({self.start_idx}, {self.end_idx})"

    @property
    def start_idx(self):
        return self._start_idx

    @property
    def end_idx(self):
        return self._end_idx
