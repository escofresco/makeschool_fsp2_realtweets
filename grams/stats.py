# grams.stats
#!/usr/bin/env python3
"""Module for generic statistics.

Stochastic sampling from a weighted frequency distribution can suck.

Example:
    distro = FreqDist([(0, 234), (1, 37), (2, 343)])
    sample = Sample(distro)

Attributes:
    FreqDist: A data structure for managing generic two-dimensional data.
    Sample: Randomly choose an element in a FreqDist in constant time.
"""
from array import array
from collections import defaultdict, namedtuple
from fractions import Fraction
from functools import lru_cache
from random import choice, random, randrange
from typing import Hashable, Iterable

from dit.divergences import jensen_shannon_divergence
from dit import Distribution
import numpy as np

from .online import Avg, Var
from .root_exceptions import *
from .utils import binsearch

__all__ = ["Distro", "FreqDist", "Sample"]


class Distro:
    """A datastructure for managing two-dimensional data, such as x,y values or
    discrete inputs and outputs.`Google Python
    Style Guide`_

    Args:
        bins: Can be any iterable object (such as dict() or list()). But if
            it isn't hashable, must be subscriptable, and the
            element contained at every index must be susciptable with length==2.

    Attributes:
        bins: The distribution data.
        dtype: The datatype of self.bins.
        bin_dtype: The datatype of each bin in self.bins.

    Raises:
        InvalidDataTypeError: bins is a string.

    .. _Google Python Style Guide:
       http://google.github.io/styleguide/pyguide.html
    """

    __slots__ = ("bins", "dtype", "bin_dtype")

    def __init__(self, bins):

        if isinstance(bins, str):
            raise InvalidDataTypeError("bins must be a two-dimensional iterable"
                                       "object.")

        self.bins = bins
        self.dtype = type(bins)
        self.bin_dtype = type(bins.items() if type(bins) is dict else bins[0]
                             ) if len(bins) else None

    def __repr__(self):
        return "\n" + "\n".join(f"{attr}: {getattr(self, attr)}"
                                for attr in FreqDist.__slots__
                                if hasattr(self, attr))

    def __len__(self):
        return len(self.bins)


class FreqDist(Distro):
    """A datastructure for managing the frequency distribution of types (words
    or other linguistic units). Optimizations here are heavily influenced by
    `Zipf's law`_, which input data is assumed to follow, by default.

        f(w) = C / r(w)^a or f(w) ∝ 1 / r(w)

        f(w): Frequency of word *w*
        r(w): Rank of the Frequency of Word *w*
        C: Number of unique words

    This predicts that for a=1, C=60,000:
        * Most frequent word occurs 60k times
        * Second most frequent word occurs 30k times
        * Third most frequent word occurs 20k times
        * And there's a long tail of 80k words with frequencies between 1.5
        and 0.5 occurences.
    This means that if a word has not been seen before, its probability is close
    to 1/N.

    Args:
        tokens_freqs: Can be any iterable object with tokens (any linguistic
            unit) mapped to their frequency. Must be homogeneous.
        sort_data: Set this to False if sorting data is unnecessary, like if
            you anticipate that searching for tokens will be infrequent.

    Raises:
        InvalidDataTypeError: The token_freqs passed to the constructor isn't
            an accepted data type.

    .. _Zipf's law:
        http://compling.hss.ntu.edu.sg/courses/hg3051/pdf/HG3051-lec05-stats.pdf
    """
    __slots__ = ("tokens_freqs", "sort_data", "types_freqs", "type_dtype",
                 "min_freq", "lowest_rank_types", "max_freq",
                 "highest_rank_types", "max_type_len", "min_type_len", "online_mean_freq",
                 "online_freq_var", "token_count", "type_count")

    def __init__(self, tokens_freqs, sort_data=True):
        self.sort_data = sort_data
        if isinstance(tokens_freqs, dict):
            pass
        elif isinstance(tokens_freqs, Iterable) and hasattr(
                tokens_freqs, "__getitem__"):
            if self.sort_data:
                ## when initialized with something iterable, like a tuple,
                ## sort and reassign as the same datatype
                try:
                    tokens_freqs = FreqDist.cast(sorted(tokens_freqs),
                                                 tokens_freqs)
                except TypeError as e:
                    InvalidTokenDatatype("Data passed to constructor must be "
                                         f"homogeneous.\n{e}")
        else:
            ## tokens_freqs isn't a valid type
            raise InvalidDataTypeError("tokens_freqs must be a dictionary or "
                                       "Iterable")
        # cast values yielded from self._make_table(<>) to the same type as
        # tokens_freqs
        tokens_freqs = FreqDist.cast(self._make_table(tokens_freqs),
                                     tokens_freqs)
        super().__init__(tokens_freqs)

    def _make_table(self, tokens_freqs):
        """Preprocessing step: Goes through a two-dimensional iterable,
        gathering statistics along the way. If the data is guaranteed to be
        sorted (self.sort_data is True), duplicate tokens are removed with their
        associated frequencies added together. Tokens and frequencies are
        validated along the way.

        Args:
            tokens_freqs: This is the first parameter passed to this classes
                constructor. It may have been sorted if self.sort_data is True.

        Yields:
            tuple[Hashable, int]: The next valid token, frequency tuple in
                tokens_freqs

        Raises:
            InvalidTokenDatatype: Token data type is inconsistent
            InvalidTokenError: Token isn't hashable
            InvalidFrequencyError: Frequency isn't an integer
            ImproperTupleFormatError: Length of token tuple is inconsistent with
                other tokens
            ImproperDataFormatError: Length of token is inconsistent with other
                tokens
        """
        self._init_instance_vars()
        last_token = None  # last token seen
        running_sum_freq = 0  # sum of frequencies for duplicates of a token

        for i, (token, freq) in enumerate(tokens_freqs.items(
        ) if type(tokens_freqs) is dict else tokens_freqs):
            ## go through tokens_freqs, constructing a cleaned and validated
            ## copy
            # bin type is tuple if tokens_freqs is dict, otherwise
            # the actual type is the value at tokens_freqs[i]
            bin_dtype = (tuple if type(tokens_freqs) is dict else type(
                tokens_freqs[i]))
            self.max_type_len = max(self.max_type_len, len(token))
            self.min_type_len = min(self.min_type_len, len(token))

            ##
            # Assignment
            ##
            if self.type_dtype is None:
                ## this is almost certainly the first iteration
                ## assignment is safe
                self.type_dtype = type(token)

            ##
            # Validation
            ##
            if self.type_dtype is not type(token):
                raise InvalidTokenDatatype("Data must be homogeneous.")

            if not isinstance(token, Hashable):
                raise InvalidTokenError(f"Token {token} must be hashable.")

            if not isinstance(freq, int):
                raise InvalidFrequencyError(
                    f"Frequency {freq} must be an integer.")

            if self.min_type_len != self.max_type_len:
                if self.type_dtype is str:
                    pass
                elif self.type_dtype is tuple:
                    raise ImproperTupleFormatError(
                        f"When using a sequence, token/type length must be "
                        "homogeneous, but found "
                        f"conflicting lengths {len(token)} and "
                        f"{self.max_type_len}")
                else:
                    raise ImproperDataFormatError(
                        f"When using a sequence, token/type length must be "
                        "homogeneous, but found "
                        f"conflicting lengths {len(token)} and "
                        f"{self.max_type_len}")
            ##
            # Logic
            ##
            if self.sort_data and type(tokens_freqs) is not dict:
                ## since tokens_freqs is guaranteed to be sorted, remove
                ## duplicate tokens and add frequencies together
                if token != last_token:
                    ## this token hasn't been seen before
                    # update instance vars since prior duplicate has been
                    # consolidated
                    if last_token is not None:
                        ## last_token has been assigned something valid
                        self._update_instance_vars(last_token, running_sum_freq)

                        yield FreqDist.cast((last_token, running_sum_freq),
                                            bin_dtype)

                    # update with current since we're no longer checking for
                    # the old duplicate
                    last_token = token
                    running_sum_freq = freq
                else:
                    ## combine frequencies, this token is a duplicate
                    running_sum_freq += freq

                if i == len(tokens_freqs) - 1 and token == last_token:
                    # yield the final vals, which are guaranteed to not be
                    # followed by a duplicate
                    self._update_instance_vars(last_token, running_sum_freq)

                    yield FreqDist.cast((last_token, running_sum_freq),
                                        bin_dtype)

            else:
                ## we don't know if tokens_freqs is sorted or a Mapping
                self._update_instance_vars(token, freq)
                yield FreqDist.cast((token, freq), bin_dtype)

    def _init_instance_vars(self):
        """Convenience method for initializing FreqDist instnace variables"""

        self.type_dtype = None  # datatype of token

        # frequency corresponding to type(s) with lowest rank
        self.min_freq = float("inf")

        # types with the lowest rank, Zipf's law suggests this will be a big
        # number
        self.lowest_rank_types = None

        # frequency corresponding to type(s) of first rank (most frequent)
        self.max_freq = float("-inf")

        # types with first rank, Zipf's law suggests this will be small, if not
        # one.
        self.highest_rank_types = None

        self.max_type_len = float("-inf") # largest length of a type
        self.min_type_len = float("inf") # smallest length of a type
        self.online_mean_freq = Avg(
        )  # track average as we pass through tokens_freqs
        self.online_freq_var = Var(
        )  # track variance as we pass through tokens_freqs
        self.token_count = 0  # number of linguistic units
        self.type_count = 0  # number of distinct linguistic units

    def _update_instance_vars(self, token, freq):
        """Convenience method for updating initialized instance variables.
        This method doesn't contain a safety check for duplicate tokens but does
        do some validation around existing instance variables.

        Args:
            token (Any): A linguistic unit for discretizing speech.
            freq (int): The number of observed token occurences.
        """

        self.token_count += freq

        if freq < self.min_freq:
            # token has the lowest rank so far
            self.min_freq = freq
            self.lowest_rank_types = np.array([token], self.type_dtype)
        elif freq == self.min_freq:
            # token has the same rank as atleast one other item
            np.append(self.lowest_rank_types, token)

        if freq > self.max_freq:
            # token has the largest rank so far
            self.max_freq = freq
            self.highest_rank_types = np.array([token], self.type_dtype)
        elif freq == self.max_freq:
            # token has the same rank as atleast one other item
            np.append(self.highest_rank_types, token)

        self.online_mean_freq.add(freq)
        self.online_freq_var.add(freq)

        self.type_count += 1

    @staticmethod
    def cast(obj, target, default=None):
        """Cast an object to a type.

        Args:
            obj: Any object.
            target: A datatype or object with a target type.
            default: The fallback datatype (or object of target datatype) to
            cast to if target fails. An exception is raised by default though.

        Raises:
            InvalidDataTypeError: Can't cast to target datatype and default
                is None or also invalid.
        """
        try:
            target_type = target if type(target) is type else type(target)
            obj_with_same_type_as_target = target_type(obj)
        except TypeError as e:
            if default is None:
                raise InvalidDataTypeError(str(e))
            else:
                obj_with_same_type_as_target = FreqDist.cast(obj, default)
        return obj_with_same_type_as_target


# class FreqDist(Distro):
#     """Manage a distribution of data."""
#     __slots__ = ("word_to_freq", "min", "max", "max_word_len", "mean", "var",
#                  "std", "tokens", "types", "words", "probs", "freqs",
#                  "cumulative_probs")
#
#     def __init__(self, word_to_freq: Iterable):
#
#         if isinstance(word_to_freq, dict):
#             ## maintain the use of a dictionary, if that's what's passed in
#             self.word_to_freq = {**word_to_freq}
#         elif isinstance(word_to_freq, Iterable):
#             ## when initialized with something iterable, like a tuple,
#             ## sort and reassign as the same datatype
#             if hasattr(word_to_freq, "sort"):
#                 word_to_freq.sort(key=lambda x: x[0])
#             else:
#                 word_to_freq = tuple(sorted(word_to_freq, key=lambda x: x[0]))
#
#             try:
#                 self.word_to_freq = tuple(FreqDist.remove_dups(word_to_freq))
#             except ValueError as e:
#                 raise e
#         else:
#             raise TypeError("word_to_freq must be Iterable")
#
#         self.min = float("inf") # smallest frequency
#         self.max = float("-inf") # largest frequency
#         self.max_word_len = float("-inf")
#         self.mean = 0
#         self.var = 0
#         self.std = 0
#         self.tokens = 0
#         self.types = len(word_to_freq)
#         (self.words, self.probs, self.freqs,
#          self.cumulative_probs) = (zip(*self._make_distro()) if
#                                    len(word_to_freq) else (((), ), (), (), ()))
#
#     def __len__(self):
#         return self.types
#
#     def __getitem__(self, key):
#
#         if isinstance(key, str):
#             # since key is a string, we'll assume it's a word
#             try:
#                 if isinstance(self.word_to_freq, dict):
#                     # we can instantly get freq
#                     return self.word_to_freq[key]
#                 idx = self.search_word(key)
#                 if idx > -1:
#                     return self.word_to_freq[idx][1]
#                 raise KeyError(f"{key} doesn't exist")
#             except (KeyError) as e:
#                 raise e
#         elif isinstance(key, slice):
#             return tuple(
#                 FreqDist.to_named_tuple((self.words[i], self.probs[i],
#                                        self.freqs[i],
#                                        self.cumulative_probs[i]))
#                 for i in range(key.start, key.stop, key.step))
#         elif not isinstance(key, int):
#             raise ValueError(f"{key} must be str or int.")
#
#         if key >= len(self.words) or key < 0:
#             raise IndexError(f"{key} not in [0,{len(self.words)})")
#         return FreqDist.to_named_tuple(
#             (self.words[key], self.probs[key], self.freqs[key],
#              self.cumulative_probs[key]))
#
#     def __contains__(self, key):
#         try:
#             _ = self[key]
#             return True
#         except:
#             return False
#
#     def __iter__(self):
#         yield from (FreqDist.to_named_tuple(t) for t in zip(
#             self.words, self.probs, self.freqs, self.cumulative_probs))
#
#     def __next__(self):
#         return next(self)
#
#     def __setitem__(self, key):
#         raise TypeError("FreqDist doesn't support assignment.")
#
#     def __repr__(self):
#         return "\n".join(f"{attr}: {getattr(self, attr)}"
#                          for attr in FreqDist.__slots__)
#
#     @staticmethod
#     def to_named_tuple(t):
#         return namedtuple("Item",
#                           ("words", "prob", "freq", "cumulative_prob"))(*t)
#
#     @staticmethod
#     def remove_dups(sorted_wordfreqs):
#         # [1,1,2] -> [1,2]; [1,1] -> [1,]
#         last_word = None
#         sum_freq = None
#
#         for i, (word, freq) in enumerate(sorted_wordfreqs):
#
#             try:
#                 FreqDist.is_valid_wordfreq(word, freq)
#             except ValueError as e:
#                 raise e
#
#             if word != last_word:
#                 if None not in {last_word, sum_freq}:
#                     # both last_word and sum_freq and been set to values in
#                     # sorted_wordfreqs
#                     yield last_word, sum_freq
#                 last_word = word
#                 sum_freq = freq
#             else:
#                 sum_freq += freq
#
#             if i == len(sorted_wordfreqs) - 1 and word == last_word:
#                 # if this is the last iteration and this word is the same as
#                 # the last, yield
#                 yield last_word, sum_freq
#
#     @staticmethod
#     def is_valid_wordfreq(word, freq):
#         """Validate word and frequency."""
#
#         if not (isinstance(word, str)):
#             # words must be strings
#             raise ValueError(f"word {word} must be a string")
#
#         if not (freq is not None and isinstance(freq, int) and freq != 0):
#             # 0s and anything but integers isn't allowed as a frequency
#             raise ValueError(
#                 f"The frequency {freq} for {word} must be an integer and "
#                 "cannot be zero")
#         return True
#
#     @lru_cache(128)
#     def search_word(self, word):
#         """binary search for word, return the index if found, -1 otherwise"""
#         return binsearch(self.word_to_freq, word, lambda x: x[0])
#
#     def get(self, key, default=None):
#         """replicate the behavior of dict.get(k)"""
#         try:
#             return self[key]
#         except (IndexError, ValueError, KeyError) as e:
#             return default
#
#     def all_word_probs_padded(self, word_size):
#         """return a list of all words and their corresponding probabilities where
#         words have been right-padded with spaces to make their size equal to
#         word_size."""
#
#         for word, freq in (self.word_to_freq.items() if isinstance(
#                 self.word_to_freq, dict) else self.word_to_freq):
#             yield (word + " " * (word_size - len(word)),
#                    Fraction(freq, self.tokens))
#
#     def _make_distro(self):
#         freq_to_words = defaultdict(list)  # inversion of self.word_to_freq
#         online_var = Var()
#         cumulative = 0
#
#         for word, freq in (self.word_to_freq.items() if isinstance(
#                 self.word_to_freq, dict) else self.word_to_freq):
#             try:
#                 _ = FreqDist.is_valid_wordfreq(word, freq)
#             except ValueError as e:
#                 raise e
#             # map frequencies to words
#             self.min = min(self.min, freq)
#             self.max = max(self.max, freq)
#             self.max_word_len = max(self.max_word_len, len(word))
#             self.tokens += freq
#             freq_to_words[freq].append(word)
#
#         for freq in range(self.min, self.max + 1):
#             # go through word counts in order
#             if freq in freq_to_words:
#                 words = tuple(freq_to_words[freq])
#                 prob = Fraction(freq * len(words), max(self.tokens, 1))
#                 cumulative += prob
#                 online_var.add(freq)
#                 yield words, prob, freq, cumulative
#
#         self.mean = online_var.mean
#         self.var = online_var.var()
#         self.std = online_var.std()
#
#     def similarity(self, other_distro):
#         """TIME EXPENSIVE
#         return the jensen-shannon distance of two distributions represented as
#         Disto(). this function is commutative
#         """
#
#         if 0 in {self.tokens, other_distro.tokens}:
#             # self or other_distro is empty, no distance exists
#             return
#         biggest_word_len = max(self.max_word_len, other_distro.max_word_len)
#
#         # every word is padded with spaces (to make them equal length) and
#         # assigned a probability of being chosen
#         word_probs = self.all_word_probs_padded(biggest_word_len)
#         other_word_probs = other_distro.all_word_probs_padded(biggest_word_len)
#         return jensen_shannon_divergence(
#             (Distribution(*zip(*word_probs)),
#              Distribution(*zip(*other_word_probs))))
#
#     def remove_word(self, word):
#         freq = self[word]


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

    Args:
        distribution (Distro): Any Distro object accepted.
    """
    __slots__ = ("distribution", "n", "alias", "prob")

    def __init__(self, distribution: Distro):
        self.distribution = distribution
        self.n = len(distribution)
        self.alias, self.prob = self._make_table()

    def __repr__(self):
        return "\n".join(f"{'-'*20}\n{attr}:\n{getattr(self, attr)}"
                         for attr in Sample.__slots__)

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
            # this would only occur if there's numerical instability
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
