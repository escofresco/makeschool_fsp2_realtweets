from collections import namedtuple
from fractions import Fraction
from random import randrange
import unittest

import numpy as np

from grams.online import Avg, Var


def avg(array):
    return sum(array) / len(array)


class OnlineTestSuite(unittest.TestCase):
    def setUp(self):
        self.online_avg = Avg()
        self.normal_large_array = [randrange(i) for i in range(1, 100001)]

    def test_online_variance_uniform_decimal(self):
        variance = Var()
        array = tuple(Fraction(randrange(1001), 100)
                      for _ in range(100000))  # variance =~ 1/12, mean =~ 1/2
        array_as_floats = tuple(map(float, array))
        expected_stats = self.expected_stats(array_as_floats)
        for num in array:
            variance.add(num)
        self.make_online_variance_asserts(expected_stats, variance)

    def test_online_variance_uniform_natural(self):
        variance = Var()
        array = tuple(randrange(1001) for _ in range(100000))
        expected_stats = self.expected_stats(array)
        for num in array:
            variance.add(num)
        self.make_online_variance_asserts(expected_stats, variance)

    @unittest.skip("Remove will likely get removed.")
    def test_online_variance_uniform_remove(self):
        variance = Var()
        array = [Fraction(randrange(101), 100) for _ in range(1000)]
        array_as_floats = list(map(float, array))
        for num in array:
            variance.add(num)

        for i in range(1000, -1, -100):
            for _ in range(100):
                variance.remove(array.pop())
            expected_stats = self.expected_stats(array_as_floats)
            with self.subTest(i=i):
                print(i)
                self.make_online_variance_asserts(expected_stats, variance)

    def test_norm_exclusively_adds(self):
        expected = avg(self.normal_large_array)
        for val in self.normal_large_array:
            self.online_avg.add(val)
        self.assertEqual(expected, round(float(self.online_avg), 5))

    def test_norm_exclusively_removes(self):
        pass

    def make_online_variance_asserts(self, expected_stats, variance):
        self.assertAlmostEqual(expected_stats.popvar, float(variance.var()), 3)
        self.assertAlmostEqual(expected_stats.samplevar,
                               float(variance.var(ddof=1)))
        self.assertAlmostEqual(expected_stats.popstd, float(variance.std()), 3)
        self.assertAlmostEqual(expected_stats.samplestd,
                               float(variance.std(ddof=1)))
        self.assertAlmostEqual(expected_stats.mean, float(variance.mean))

    @staticmethod
    def expected_stats(array):
        Expected = namedtuple("Expected",
                              "popvar samplevar mean popstd samplestd ")
        return Expected(np.var(array), np.var(array, ddof=1), np.mean(array),
                        np.std(array), np.std(array, ddof=1))
