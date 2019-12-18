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
from collections import Counter, defaultdict, namedtuple
from fractions import Fraction
from functools import lru_cache
from random import choice, random, randrange
from typing import Hashable, Iterable

from dit.divergences import jensen_shannon_divergence
from dit import Distribution
import numpy as np

from .online import Avg, Var
from .root_exceptions import *
from .termgraph import showgraph
from .utils import binsearch

__all__ = ["Distro", "FreqDist", "Sample"]


class Distro:
    """A datastructure for managing two-dimensional data, such as x,y values or
    discrete inputs and outputs.`Google Python
    Style Guide`_

    Args:
        bins: Can be any iterable object (such as dict() or list()). But if
            it isn't hashable, it must be subscriptable. Every column of bins
            must also be independently homogeneous.

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
            raise InvalidDataTypeError("Strings are accepted. bins must be a "
                                       "two-dimensional iterable object.")

        if (not issubclass(type(bins), dict) and
                not hasattr(bins, "__getitem__")):
            ## bins is neither the subclass of dict nor subscriptable
            raise InvalidDataTypeError("All dict subclasses and most "
                                       "subscriptable objects are accepted, but"
                                       f"\n{bins} is none of these.")
        self.bins = bins
        self.dtype = type(bins)

        # assign dict_items if bins is a subclass of dict, otherwise bins
        # is expected to accept integers as indexes.
        self.bin_dtype = type(bins.items(
        ) if issubclass(type(bins), dict) else bins[0]) if len(bins) else None

    def __repr__(self):
        return "\n" + "\n".join(f"{attr}: {getattr(self, attr)}"
                                for attr in FreqDist.__slots__
                                if hasattr(self, attr))

    def __len__(self):
        return len(self.bins)

    def find(self, target, key=None, is_sorted=False):
        """Search through self.bins for val, with an extra speed-up if
        the data for self.bins is already sorted.

        Args:
            target: If target is callable, it's expected to accept two arguments.
                    The first is an element being looked at, the second is a
                    comparison function.
                    If target isn't callable, it just represents a value to
                    find.
            key: This is called on every element being searched. Expected to
                return a new element to check instead.
            is_sorted: False, by default. Passing True here indicates that
                the first column of self.bins is sorted.

        Returns:
            If target is found, either its index or corresponding key will be
            returned. Otherwise, None is returned.
        """

        if not callable(target):
            ## Since target isn't callable, it's converted to something that is.
            ## Taking a comparison function as an argument, it's expected to
            ## check that against the item being searched for, returning a bool.
            def is_match(target_val):

                def _is_match(cmp):
                    res = cmp(target_val)
                    if res is NotImplemented:
                        return False
                    return res

                return _is_match

            target = is_match(target)

        if key is None:
            key = lambda elm: elm.__eq__

        def is_match(possible_val):
            ## check if possible_val equals or contains a value equalling target.

            # map key to possible_val
            #key_map = map(lambda elm: key(elm).__eq__, (possible_val,))
            key_map = map(lambda elm: key(elm), (possible_val,))

            # return True if target returns True for any element in key_map
            return any(map(target, key_map))

        if issubclass(self.dtype, dict):
            ## dict subclass
            for cur_key, cur_val in self.bins.items():
                if is_match(cur_val):
                    return cur_key

        # elif is_sorted:
        #     ## sorted sequence
        #     ## use binary search for sorted first column
        #     idx = binsearch(self.bins, target)
        #     if idx > -1:
        #         return idx
        #
        #     ## use linear search for search for second column
        #     for i, (_, elm) in enumerate(self.bins):
        #         if type(elm) is not type(val):
        #             ## knowing the homogeneity invariant is maintained, no
        #             ## other elements in this column would be a match
        #             break
        #         if target(elm.__eq__):
        #             return i
        else:
            ## sequence
            for i, bin in enumerate(self.bins):
                if any(map(is_match, bin)):
                    ## is_match returned True for at least one element in bin
                    return i

    def show(self):
        """Visualize the data contained within this class.

        Args:
            key: An optional function can be applied across every bin.

        Raises:
            MissingDataError: self.bins is empty, so there's nothing to show.
        """
        if not len(self):
            raise MissingDataError("Whoop! It looks like there isn't any data "
                                   "to display.")

        if issubclass(type(self.bins), dict):
            labels, data = zip(*self.bins.items())
        else:
            labels, data = zip(*self.bins)

        # since termgraph uses categories, restructure data as 2d
        data = tuple((elm,) for elm in data)
        showgraph(labels=tuple(map(str, labels)), data=data)


class FreqDist(Distro):
    """A language-specific datastructure for managing the frequency distribution of types (words
    or other linguistic units). A frequency distribution represents the number
    of times each event in an experiment occurs. Data is broken into groups,
    which are referred to here as bins. Optimizations are heavily
    influenced by `Zipf's law`_, which input data is assumed to follow, unless
    otherwise specified.

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
            unit) mapped to their frequency. Must be homogeneous. This class
            does its best to maintain the original datatype that was passed in.
                Examples:
                    >>> data_to_frequency = {}
                    >>> freqdist = FreqDist(data_to_frequency)
                    >>> type(data_to_frequency) is type(freqdist.bins)
                    True

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
                 "highest_rank_types", "max_type_len", "min_type_len",
                 "online_mean_freq", "online_freq_var", "token_count",
                 "type_count")

    def __init__(self, tokens_freqs, sort_data=True):
        self.sort_data = sort_data
        if issubclass(type(tokens_freqs), dict):
            ## tokens_freqs is either a dict or a subclass of it
            # cast to generic dict
            tokens_freqs = FreqDist.cast(tokens_freqs, dict)
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

    def __getitem__(self, key):
        """Override magic method.

        Args:
            vals: Depending on the dtype of this distribution, this is a value
                for get a specific bin.

        Returns:
            value existing at the bin for key.
        """

        if issubclass(type(self.bins), dict):
            ## hashable, existing vals are expected
            if not isinstance(key, Hashable):
                raise InvalidKeyError(f"{key} must be hashable.")
            if key not in self.bins:
                raise KeyNotFoundError(f"{key} doesn't exist in this "
                                       "distribution.")
        else:
            if not isinstance(key, int):
                raise InvalidIndexError(f"{key} must be an integer.")

            if key >= len(self):
                raise IndexNotFoundError(f"{key} isn't a valid index for "
                                         "this distribution, with length "
                                         f"{self.token_count}.")
        return self.bins[key]

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
        ) if issubclass(type(tokens_freqs), dict) else tokens_freqs):
            ## go through tokens_freqs, constructing a cleaned and validated
            ## copy
            # bin type is tuple if tokens_freqs is dict, otherwise
            # the actual type is the value at tokens_freqs[i]
            bin_dtype = (tuple if issubclass(type(tokens_freqs), dict) else
                         type(tokens_freqs[i]))
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
            if self.sort_data and issubclass(type(tokens_freqs), dict):
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

        self.max_type_len = float("-inf")  # largest length of a type
        self.min_type_len = float("inf")  # smallest length of a type
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

    def prob(self, bin):
        """Lookup the type and frequency at bin and return it's probability.

        Args:
            bin: The index of a type/frequency item.

        Returns:
            frequency / tokens_count, the probability of type being observed.
        """
        try:
            prob = Fraction(self[bin][1], self.token_count)
        except Error:
            raise
        return prob

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


class Sample:
    """Use Vose's Alias Method to sample from a non-uniform discrete
    distribution. there's no tradeoff between this and roulette wheel selection;
    space and time complexity for initialization are equivalent, but time
    complexity of generation for Vose's Alias Method is theoretically
    optimal.::
        time:
                initialization: Θ(n)
                sampling: Θ(1)
        space:  Θ(n)

    Args:
        distribution (FreqDist): Any FreqDist object accepted.
    """
    __slots__ = ("distribution", "n", "alias", "prob")

    def __init__(self, distribution: FreqDist):
        self.distribution = distribution
        self.n = len(distribution)
        self.alias, self.prob = self._make_table()

    def __repr__(self):
        return "\n".join(f"{'-'*20}\n{attr}:\n{getattr(self, attr)}"
                         for attr in Sample.__slots__)

    def _make_table(self):
        """Preprocessing step: go through distribution and build up prob/alias
        table. time: best = worst = Θ(n)
        """

        # make temp alias and prob arrays
        alias = [0 for _ in range(self.n)]  #array("L", [0] * self.n)
        prob = [0 for _ in range(self.n)]  #array("d", [0] * self.n)

        # create worklists for managing the construction of alias and prob
        small = []  #array("L")
        large = []  #array("L")

        for i in range(self.n):
            # build worklists
            # time: Θ(n)
            p_i = self.distribution.prob(i) * self.n  # scale prob by n
            small.append(i) if p_i < 1 else large.append(i)

        while len(small) and len(large):
            # time: O(n)
            l = small.pop()
            g = large.pop()
            p_l = self.distribution.prob(l)
            prob[l] = p_l
            alias[l] = g
            p_g = (self.distribution.prob(g) + p_l) - 1
            small.append(g) if p_g < 1 else large.append(g)

        while len(large):
            # time: at most O(n)
            prob[large.pop()] = 1

        while len(small):
            # this would only occur because of numerical instability
            prob[small.pop()] = 1

        return tuple(alias), tuple(prob)

    def rand(self):
        """Generation step: calculate the index to an element.
        time: best = worst = Θ(1)

        Returns:
            The integer index to a selected element in self.distribution.
        """
        i = randrange(self.n)
        is_heads = random() < self.prob[i]
        if is_heads:
            return i
        return self.alias[i]

    def randword(self):
        """Select a type from self.distribution using a random index.

        Returns:
            An object representing a randomly selected element.

        """
        return self.distribution[self.rand()]

    def randwords(self, n):
        """Select random words.

        Yields:
            An object representing a randomly selected element.
        """
        while n > 0:
            yield self.randword()
            n -= 1
