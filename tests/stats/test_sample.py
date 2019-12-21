from collections import Counter, defaultdict, namedtuple, UserDict, UserList
from dataclasses import dataclass
from fractions import Fraction
from math import floor
from random import randint, randrange
import unittest

import numpy as np

from grams.online import Avg, Var
from grams.root_exceptions import *
from grams.stats import Distro, FreqDist, Sample
from grams.utils import randints


@dataclass
class TD:
    COLOR_TUPLE_FREQS = (("orange", 10), ("yellow", 16), ("green", 8),
                         ("cyan", 20), ("grey", 8), ("blue", 8), ("pink", 10))
    COLOR_TUPLE_PROBS = (("orange", Fraction(1, 8)), ("yellow", Fraction(1, 5)),
                         ("green", Fraction(1, 10)), ("cyan", Fraction(1, 4)),
                         ("grey", Fraction(1, 10)), ("blue", Fraction(1, 10)),
                         ("pink", Fraction(1, 8)))


class SampleTestSuite(unittest.TestCase):

    def test_init(self):
        uniform_freqdist = FreqDist([("apple", 2), ("banana", 2),
                                     ("orange", 2)])
        sample = Sample(uniform_freqdist)
        self.assertIs(sample.distribution, uniform_freqdist)
        self.assertEqual(sample.n, 3)

    def test_preprocessing(self):
        freqdist = FreqDist(TD.COLOR_TUPLE_FREQS, sort_data=False)
        sample = Sample(freqdist)

        self.assertEqual(sample.n, 7)

        expected_alias = (1, None, 3, 1, 3, 3, 3)
        expected_prob = (Fraction(7, 8), Fraction(1, 1), Fraction(7, 10),
                         Fraction(29, 40), Fraction(7, 10), Fraction(7, 10),
                         Fraction(7, 8))
        self.assertEqual(expected_alias, sample.alias)
        self.assertEqual(expected_prob, sample.prob)

    def test_generation_from_sequence(self):
        N_SAMPLES = 10000
        freqdist = FreqDist(TD.COLOR_TUPLE_FREQS, sort_data=False)
        sample = Sample(freqdist)

        self.assertEqual(sample.n, 7)

        sample_counts = Counter()
        for _ in range(N_SAMPLES):
            sample_counts[sample.randbin()] += 1

        samples = FreqDist(sample_counts)

        self.assertLess(freqdist.similarity(samples), 0.05)

    def test_generation_from_mapping(self):
        N_SAMPLES = 10000
        freqdist = FreqDist(dict(TD.COLOR_TUPLE_FREQS), sort_data=False)
        sample = Sample(freqdist)

        self.assertEqual(sample.n, 7)

        sample_counts = Counter()
        for _ in range(N_SAMPLES):
            sample_counts[sample.randbin()] += 1

        samples = FreqDist(sample_counts)

        self.assertLess(freqdist.similarity(samples), 0.05)


class TestTestData(unittest.TestCase):

    def test_color_probs_sum_to_one(self):
        self.assertEqual(1., sum(elm[1] for elm in TD.COLOR_TUPLE_PROBS))
