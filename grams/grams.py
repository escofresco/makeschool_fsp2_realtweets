#!python

from __future__ import division, print_function  # Python 2 and 3 compatibility
from array import array
from collections import Counter, defaultdict, deque, namedtuple
from dataclasses import make_dataclass
from fractions import Fraction
from functools import wraps
from os import devnull
from random import random, choice
import sys
from typing import Iterable, Optional, Tuple, Union

from coverage import Coverage, CoverageData
from nltk import pos_tag, sent_tokenize, word_tokenize

from .online import Var
from .root_exceptions import *
from .stats import FreqDist, Sample
from .termgraph import showgraph
from .utils import (binsearch, invert_dict, LogMethodCalls,
                    merge_data_containing_ints, p)

__all__ = ["Histogram", "Listogram", "Dictogram"]


class Gram(FreqDist):  #, metaclass=LogMethodCalls, logs_size=4):
    """This is a generic histogram which holds a distribution of data, which
    have been clumped into bins by the parent class, FreqDist. Bins are aligned
    in two dimensions, data and frequency, where frequency is calculated from
    data.

    Args:
        data: This represents the input to a probability distribution function.
    """
    __slots__ = ("data_frequency", "sampler")

    def __init__(self, data_frequency):
        super().__init__(data_frequency)
        self.sampler = Sample(self)

    def similarity(self, other):
        return super().similarity(other)

    def freq(self, bin):
        """Find the frequency of bin.
        Args:
            bin: The object to find the frequency of.
        """
        try:
            freq = super().freq(bin)
        except (KeyError, IndexError):
            freq = None
        except:
            raise
        return 0 if freq is None else freq

    def sample(self):
        return self.sampler.randbin()

    @staticmethod
    def bin_search(array, prob):
        """Search for the lowest matching probability in array consisting of
        namedtuples. this is the sample component of roulette wheel sampling"""
        lo = 0
        hi = len(array) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if (array[mid].cumulative <= prob and
                (mid == len(array) - 1 or array[mid + 1].cumulative > prob)):
                return mid
            if array[mid].cumulative < prob:
                lo = mid + 1
            else:
                hi = mid
        return None

    @staticmethod
    def sents(block_text):
        """Convert multiline text into an array of sentences.

        Args:
            block_text (str): Can be any string.

        Yields:
            The next sentence as a tuple of words.
        """
        for sentence in sent_tokenize(block_text):
            yield word_tokenize(sentence)

    @staticmethod
    def pos_sents(block_text):
        """Convert multiline text into an array of sentences, with each word tagged
        with part of speech.

        Args:
            block_text (str): Can be any string.

        Yields:
            The next sentence as a tuple of word and pos_tag tuples.
        """
        for sentence in sent_tokenize(block_text):
            yield pos_tag(word_tokenize(sentence))


class Histogram(Gram, metaclass=LogMethodCalls, logs_size=4):

    __slots__ = ("corpus", "tokens_freqs", "sampler")

    def __init__(self, corpus=None, tokens_freqs=None, use_pos_tags=False):
        """Takes text or a pregenerated histogram as input."""
        default_dtype = dict
        if tokens_freqs is None:
            dtype = default_dtype
            if corpus is None:
                super().__init__(default_dtype())
            else:
                super().__init__(default_dtype(self._make_token_freq(corpus, use_pos_tags=use_pos_tags)))
        else:
            dtype = type(tokens_freqs)
            super().__init__(tokens_freqs)
        self.sampler = Sample(self)

    def _make_token_freq(self, corpus, use_pos_tags):
        yield from Gram.pos_sents(corpus) if use_pos_tags else Gram.sents(corpus)

    def frequency(self, token):
        self.rebuild_with_latent_wordcounts()
        try:
            return super().freq(token)
        except:
            raise

    def sample(self):
        """Return a word from this histogram, randomly sampled by weighting
        each word's probability of being chosen by its observed frequency."""
        self.rebuild_with_latent_wordcounts()
        try:
            return super().sampler.randbin()
        except:
            raise


class Listogram(Gram, metaclass=LogMethodCalls, logs_size=4):
    """This is a histogram that strictly enforces the use of tuples where
    applicable.

    Args:
        token_list: A list of tokens to generate types and frequencies from.
    """

    __slots__ = ("tokens_list", "tmp_token_freq", "sampler")

    def __init__(self, tokens_list=None):
        # hold a temporary array as new (word,counts) get added.
        # add these to new Listogram object when add_count method calls are
        # finished.
        self.tmp_token_freq = []

        if tokens_list is not None:
            super().__init__(tuple(Counter(tokens_list).items()))
        else:
            super().__init__(())

    def add_count(self, token, count=1):
        """Increase frequency count of given word by given count amount."""
        # build temp array; duplicates are handled later
        self.tmp_token_freq.append((token, count))

    def frequency(self, token):
        self.rebuild_with_latent_wordcounts()
        try:
            return super().freq(token)
        except:
            raise

    def __contains__(self, token):
        """Return boolean indicating if given word is in this histogram."""
        self.rebuild_with_latent_wordcounts()
        return super().__contains__(token)

    def _index(self, target):
        """Return the index of entry containing given target word if found in
        this histogram, or None if target word is not found."""
        self.rebuild_with_latent_wordcounts()
        return self.index_of(target)

    def index_of(self, target):
        """Return the index of entry containing given target word if found in
        this histogram, or None if target word is not found."""
        self.rebuild_with_latent_wordcounts()
        return self.find(target)

    def sample(self):
        """Return a word from this histogram, randomly sampled by weighting
        each word's probability of being chosen by its observed frequency."""
        self.rebuild_with_latent_wordcounts()
        try:
            return self.sampler.randbin()
        except:
            raise

    def rebuild_with_latent_wordcounts(self):
        """Reconstruct this object if last method called was add_count."""
        if len(self._logs_) > 2 and self._logs_[-1 - 2].name == "add_count":
            # if the most recent class or instance method call wasn't add_count,
            # re-initialize current object.
            super().__init__(
                merge_data_containing_ints(self.bins, self.tmp_token_freq))
            self.sampler = Sample(self)


class Dictogram(Gram, metaclass=LogMethodCalls, logs_size=4):
    """This is a histogram that strictly enforces the use of dict types where
    applicable.

    Args:
        tokens_list: A list of tokens to be consolidated into types and
            their corresponding freqencies.
    """

    __slots__ = ("tmp_token_freq", "tokens_list", "sampler")

    def __init__(self, tokens_list=None):

        # Temporarily hold the (word,count) added from add_count, which will
        # be added to a new distribution as part of a new Dictogram
        self.tmp_token_freq = defaultdict(int)

        if tokens_list is not None:
            super().__init__(Counter(tokens_list))
        else:
            super().__init__({})

    def __contains__(self, token):
        """Return boolean indicating if given word is in this histogram."""
        self.rebuild_with_latent_wordcounts()
        return super().__contains__(token)

    def items(self):
        self.rebuild_with_latent_wordcounts()
        return self.bins.items()

    def add_count(self, token, count=1):
        """TIME EXPENSIVE: must call super().__init__() every time
        Increase frequency count of given word by given count amount."""
        self.tmp_token_freq[token] += count

    def frequency(self, token):
        """Return frequency count of given word, or 0 if word is not found."""
        self.rebuild_with_latent_wordcounts()
        try:
            return super().freq(token)
        except:
            raise

    def sample(self):
        """Return a word from this histogram, randomly sampled by weighting
        each word's probability of being chosen by its observed frequency."""
        self.rebuild_with_latent_wordcounts()
        try:
            return self.sampler.randbin()
        except:
            raise

    def rebuild_with_latent_wordcounts(self):
        """Reconstructs object if last method called was add_count."""
        if len(self._logs_) > 2 and self._logs_[-1 - 2].name == "add_count":
            # if the most recent class or instance method call wasn't add_count,
            # re-initialize current object.
            super().__init__(
                merge_data_containing_ints(self.bins, self.tmp_token_freq))
            self.sampler = Sample(self)


class Fuzzygram(Gram, metaclass=LogMethodCalls, logs_size=4):
    pass


class Covergram(Gram, metaclass=LogMethodCalls, logs_size=4):
    """Takes a coverage report and generates a histogram of modules and their
    corresponding code coverage as a percent."""
    __slots__ = ("coverage", "coverage_data", "sampler", "_logs_")

    def __init__(self, filepath):
        self.coverage = Coverage(data_file=filepath)
        self.coverage.load()
        self.coverage_data = CoverageData()
        self.coverage_data.read_file(filepath)
        module_to_coverage = tuple(self.as_module_to_coverage())
        super().__init__(module_to_coverage)
        self.sampler = Sample(self)

    def as_module_to_coverage(self):
        """Give next (<module name>, <percent module code coverage>) in
        coverage file, a format that can be passed to the FreqDist constructor"""
        original_stdout = sys.stdout  # store original stdout state

        # temporarily suppress standard output since calling
        # coverage.Coverage.report causes yucky output
        sys.stdout = open(devnull, "w")
        line_counts = self.coverage_data.line_counts()
        total_lines = sum(count for count in line_counts.values())
        for module in self.coverage_data._lines:

            # assign coverage as a rational number between 0 and 100
            module_coverage = Fraction(self.coverage.report(module))

            yield module, int(module_coverage)
        sys.stdout = original_stdout  # restore standard output

    def frequency(self, module):
        return self.get(module, 0)

    def sample(self):
        raise self.sampler.rand_word()


class Fuzzygram(Gram, metaclass=LogMethodCalls, logs_size=4):
    pass
