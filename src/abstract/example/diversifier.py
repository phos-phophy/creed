import random
from typing import Dict, NamedTuple, Sequence


class DiversifierConfig(NamedTuple):
    replace_prob: float = 0
    noise_prob: float = 0
    mapping: Dict[str, Sequence[str]] = {}

    def validate(self):
        if self.replace_prob + self.noise_prob > 1.0:
            raise ValueError("The sum of probabilities cannot be greater than 1!")
        if self.replace_prob < 0:
            raise ValueError("Probability of replacement cannot be less than 0!")
        if self.noise_prob < 0:
            raise ValueError("Probability of noise cannot be less than 0!")

    @property
    def active(self):
        return self.replace_prob > 0 or self.noise_prob > 0


class Diversifier:
    def __init__(self, tokenizer, config: DiversifierConfig):

        config.validate()

        self._vocab = tuple(tokenizer.get_vocab().keys()) if tokenizer else ()
        self._replace = config.replace_prob
        self._noise = config.noise_prob + config.replace_prob
        self._mapping = config.mapping

        self._config = config

    def __getitem__(self, item):
        chance = random.random()
        if chance < self._replace:
            return random.choice(self._mapping.get(item, [item]))
        elif chance < self._noise and self._vocab:
            return random.choice(self._vocab)
        return item

    @property
    def active(self):
        return self._config.active
