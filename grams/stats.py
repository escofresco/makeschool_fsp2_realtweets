from array import array
from collections import defaultdict, namedtuple
from fractions import Fraction
from functools import lru_cache
from random import choice, random, randrange
from typing import Iterable

from dit.divergences import jensen_shannon_divergence
from dit import Distribution

from grams.online import Var
from grams.utils import binsearch


class Distro:
    """Manage a distribution of data."""
    __slots__ = ("word_to_freq", "min", "max", "max_word_len", "mean", "var",
                 "std", "tokens", "types", "words", "probs", "freqs",
                 "cumulative_probs")

    def __init__(self, word_to_freq: Iterable):

        if isinstance(word_to_freq, dict):
            self.word_to_freq = {**word_to_freq}
        elif isinstance(word_to_freq, Iterable):

            if hasattr(word_to_freq, "sort"):
                word_to_freq.sort(key=lambda x: x[0])
            else:
                word_to_freq = tuple(sorted(word_to_freq, key=lambda x: x[0]))

            try:
                self.word_to_freq = tuple(Distro.remove_dups(word_to_freq))
            except ValueError as e:
                raise e
        else:
            raise TypeError("word_to_freq must be Iterable")

        self.min = float("inf")
        self.max = float("-inf")
        self.max_word_len = float("-inf")
        self.mean = 0
        self.var = 0
        self.std = 0
        self.tokens = 0
        self.types = len(word_to_freq)
        (self.words, self.probs, self.freqs,
         self.cumulative_probs) = (zip(*self._make_distro()) if
                                   len(word_to_freq) else (((), ), (), (), ()))

    def __len__(self):
        return self.types

    def __getitem__(self, key):

        if isinstance(key, str):
            # since key is a string, we'll assume it's a word
            try:
                if isinstance(self.word_to_freq, dict):
                    # we can instantly get freq
                    return self.word_to_freq[key]
                idx = self.search_word(key)
                if idx > -1:
                    return self.word_to_freq[idx][1]
                raise KeyError(f"{key} doesn't exist")
            except (KeyError) as e:
                raise e
        elif isinstance(key, slice):
            return tuple(
                Distro.to_named_tuple((self.words[i], self.probs[i],
                                       self.freqs[i],
                                       self.cumulative_probs[i]))
                for i in range(key.start, key.stop, key.step))
        elif not isinstance(key, int):
            raise ValueError(f"{key} must be str or int.")

        if key >= len(self.words) or key < 0:
            raise IndexError(f"{key} not in [0,{len(self.words)})")
        return Distro.to_named_tuple(
            (self.words[key], self.probs[key], self.freqs[key],
             self.cumulative_probs[key]))

    def __contains__(self, key):
        try:
            _ = self[key]
            return True
        except:
            return False

    def __iter__(self):
        yield from (Distro.to_named_tuple(t) for t in zip(
            self.words, self.probs, self.freqs, self.cumulative_probs))

    def __next__(self):
        return next(self)

    def __setitem__(self, key):
        raise TypeError("Distro doesn't support assignment.")

    def __repr__(self):
        return "\n".join(f"{attr}: {getattr(self, attr)}"
                         for attr in Distro.__slots__)

    @staticmethod
    def to_named_tuple(t):
        return namedtuple("Item",
                          ("words", "prob", "freq", "cumulative_prob"))(*t)

    @staticmethod
    def remove_dups(sorted_wordfreqs):
        # [1,1,2] -> [1,2]; [1,1] -> [1,]
        last_word = None
        sum_freq = None

        for i, (word, freq) in enumerate(sorted_wordfreqs):

            try:
                Distro.is_valid_wordfreq(word, freq)
            except ValueError as e:
                raise e

            if word != last_word:
                if None not in {last_word, sum_freq}:
                    # both last_word and sum_freq and been set to values in
                    # sorted_wordfreqs
                    yield last_word, sum_freq
                last_word = word
                sum_freq = freq
            else:
                sum_freq += freq

            if i == len(sorted_wordfreqs) - 1 and word == last_word:
                # if this is the last iteration and this word is the same as
                # the last, yield
                yield last_word, sum_freq

    @staticmethod
    def is_valid_wordfreq(word, freq):
        """Validate word and frequency."""

        if not (isinstance(word, str)):
            # words must be strings
            raise ValueError(f"word {word} must be a string")

        if not (freq is not None and isinstance(freq, int) and freq != 0):
            # 0s and anything but integers isn't allowed as a frequency
            raise ValueError(
                f"The frequency {freq} for {word} must be an integer and "
                "cannot be zero")
        return True

    @lru_cache(128)
    def search_word(self, word):
        """binary search for word, return the index if found, -1 otherwise"""
        return binsearch(self.word_to_freq, word, lambda x: x[0])

    def get(self, key, default=None):
        """replicate the behavior of dict.get(k)"""
        try:
            return self[key]
        except (IndexError, ValueError, KeyError) as e:
            return default

    def all_word_probs_padded(self, word_size):
        """return a list of all words and their corresponding probabilities where
        words have been right-padded with spaces to make their size equal to
        word_size."""

        for word, freq in (self.word_to_freq.items() if isinstance(
                self.word_to_freq, dict) else self.word_to_freq):
            yield (word + " " * (word_size - len(word)),
                   Fraction(freq, self.tokens))

    def _make_distro(self):
        freq_to_words = defaultdict(list)  # inversion of self.word_to_freq
        online_var = Var()
        cumulative = 0

        for word, freq in (self.word_to_freq.items() if isinstance(
                self.word_to_freq, dict) else self.word_to_freq):
            try:
                _ = Distro.is_valid_wordfreq(word, freq)
            except ValueError as e:
                raise e
            # map frequencies to words
            self.min = min(self.min, freq)
            self.max = max(self.max, freq)
            self.max_word_len = max(self.max_word_len, len(word))
            self.tokens += freq
            freq_to_words[freq].append(word)

        for freq in range(self.min, self.max + 1):
            # go through word counts in order
            if freq in freq_to_words:
                words = tuple(freq_to_words[freq])
                prob = Fraction(freq * len(words), max(self.tokens, 1))
                cumulative += prob
                online_var.add(freq)
                yield words, prob, freq, cumulative

        self.mean = online_var.mean
        self.var = online_var.var()
        self.std = online_var.std()

    def similarity(self, other_distro):
        """TIME EXPENSIVE
        return the jensen-shannon distance of two distributions represented as
        Disto(). this function is commutative
        """

        if 0 in {self.tokens, other_distro.tokens}:
            # self or other_distro is empty, no distance exists
            return
        biggest_word_len = max(self.max_word_len, other_distro.max_word_len)

        # every word is padded with spaces (to make them equal length) and
        # assigned a probability of being chosen
        word_probs = self.all_word_probs_padded(biggest_word_len)
        other_word_probs = other_distro.all_word_probs_padded(biggest_word_len)
        return jensen_shannon_divergence(
            (Distribution(*zip(*word_probs)),
             Distribution(*zip(*other_word_probs))))

    def remove_word(self, word):
        freq = self[word]


class Sample:
    """use Vose's Alias Method to sample from a non-uniform discrete
    distribution. there's no tradeoff between this and roulette wheel selection;
    space complexity and time complexity for initialization are equivalent to
    roulette wheel selection. time complexity of generation is theoretically
    optimal.
    time:
            initialization: Θ(n)
            sampling: Θ(1)
    space:  Θ(n)
    """
    __slots__ = ("distribution", "n", "alias", "prob")

    def __init__(self, distribution: Distro):
        self.distribution = distribution
        self.n = len(distribution.freqs)
        self.alias, self.prob = self._make_table()

    def __repr__(self):
        return "\n".join(f"{'-'*20}\n{attr}:\n{getattr(self, attr)}"
                         for attr in self.__slots__)

    def _make_table(self):
        """go through distribution and build up prob/alias table:
        time: best = worst = Θ(n)"""

        # make temp alias and prob arrays
        alias_arr = array("L", [0] * self.n)
        prob_arr = array("d", [0] * self.n)

        # create worklists for managing the construction of alias and prob
        small = array("L")
        large = array("L")

        for i, item in enumerate(self.distribution):
            # build worklists
            # time: Θ(n)
            cur_prob = item.prob * self.n
            small.append(i) if cur_prob < 1 else large.append(i)

        while len(small) and len(large):
            # time: O(n)
            l = small.pop()
            g = large.pop()
            p_l = self.distribution[l].prob
            p_g = (self.distribution[g].prob + p_l) - 1
            prob_arr[l] = p_l
            alias_arr[l] = g
            small.append(g) if p_g < 1 else large.append(g)

        while len(large):
            # time: at most O(n)
            prob_arr[large.pop()] = 1

        while len(small):
            """this would only occur if there's numerical instability"""
            prob_arr[small.pop()] = 1

        return tuple(alias_arr), tuple(prob_arr)

    def rand(self):
        """Give a weighted, random index.
        time: best = worst = Θ(1)"""
        i = randrange(self.n)
        is_heads = random() < self.prob[i]
        if is_heads:
            return i
        return self.alias[i]

    def randword(self):
        """Gives a random word from self.distribution."""
        return choice(self.distribution[self.rand()].words)

    def randwords(self, n):
        while n > 0:
            yield self.randword()
            n -= 1
