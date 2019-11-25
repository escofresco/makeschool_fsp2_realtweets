from collections import namedtuple, defaultdict
from dataclasses import make_dataclass
from fractions import Fraction
from functools import reduce

import numpy as np

from grams.online import Avg
from grams.utils import invert_dict


class WordData:
    __slots__ = ("words", "word_to_freq", "distribution", "about")

    def __init__(self, words):
        self.about = make_dataclass(
            "About",
            list(
                zip(("min", "max", "mean", "var", "std", "tokens", "types"),
                    [int] * 7)))(*[0] * 7)

        self.words = words
        self.about.min = float("inf")
        self.about.max = float("-inf")
        self.about.tokens = len(words)
        self.word_to_freq = defaultdict(int)
        for word in words:
            self.word_to_freq[word] += 1
        self.about.types = len(self.word_to_freq)

        self.distribution = (tuple(
            sorted(list(self._make_distribution()), key=lambda x: x.freq))
                             if len(words) else ((), (), (), ((), )))

    def _make_distribution(self):
        Item = namedtuple("Item", "prob freq cumulative_prob words")

        freq_words = tuple(
            sorted(list(
                (freq, tuple(words))
                for freq, words in invert_dict(self.word_to_freq).items()),
                   key=lambda x: x[0]))
        cumulative = 0

        freqs, words = zip(*freq_words)
        freq_ndarray = np.array(freqs)
        self.about.var = np.var(freq_ndarray)
        self.about.mean = np.mean(freq_ndarray)
        self.about.std = np.std(freq_ndarray)
        self.about.min = np.min(freq_ndarray)
        self.about.max = np.max(freq_ndarray)
        for freq, words in freq_words:
            prob = Fraction(freq * len(words), self.about.tokens)
            cumulative += prob
            yield Item(prob, freq, cumulative, tuple(sorted(words)))


no_word_data = WordData(())
one_word_data = WordData(("word", ))
small_data = WordData(
    ("the", "orange", "banana", "peach", "a", "or", "good", "tasty", "banana",
     "or", "the", "or", "tasty", "the", "orange"))
small_uniform_data = WordData(("the", "orange", "banana", "peach", "a", "or",
                               "good", "tasty", "mango", "misty"))

DISTRO_DISTANCE_THRESHOLD = 0.05
