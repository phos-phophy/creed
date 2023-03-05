import random
from typing import Dict, Tuple


class Diversifier:
    def __init__(self, tokenizer, replace_prob: float, noise_prob: float, mapping: Dict[str, Tuple[str]]):
        self._tokenizer = tokenizer
        self._replace = replace_prob
        self._noise = noise_prob + replace_prob
        self._mapping = mapping

        if replace_prob + noise_prob > 1.0:
            raise ValueError("The sum of probabilities cannot be greater than 1")

    def __getitem__(self, item):
        chance = random.random()
        if chance < self._noise:
            return random.choice(self._tokenizer.get_vocab())
        elif chance < self._replace:
            return self._mapping[item]
        return item
