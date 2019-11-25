from collections import namedtuple
from fractions import Fraction
from os.path import dirname, join
from random import uniform
import unittest

from data import one_word_data, small_data, small_uniform_data
from grams.grams import Histogram
from grams.utils import (sample_size, generate_samples, histogram_similarity,
                         map_to_binary)


class HistogramTestSuite(unittest.TestCase):
    PROJECT_ROOT = dirname(dirname(__file__))
    DIVERGENCE_THRESHOLD = 0.05

    def setUp(self):
        self.freq_to_words = {
            1: {"apple", "banana"},
            3: {"orange", "mango"},
            5: {"the"},
            6: {"a"}
        }
        self.total_words = sum(i * len(w)
                               for i, w in self.freq_to_words.items())
        self.FreqencyItem = namedtuple("Item", "freq cumulative words")
        self.cumulative_freq_list = list(self.make_cumulative())

        self.small_histogram = Histogram([
            "An apple a day will keep the doctor away.",
            "A penny saved is a penny earned.", "Martha Stewart made me a pie."
        ])
        with open(join(HistogramTestSuite.PROJECT_ROOT, "res/words")) as f:
            self.uniform_histogram = Histogram(f.readlines())

    def make_cumulative(self):
        cumulative = 0
        for freq, words in self.freq_to_words.items():
            yield self.FreqencyItem(freq, cumulative, words)
            cumulative += Fraction(freq * len(words), self.total_words)

    def test_probs_are_cumulative(self):
        # Test the cumulative list for tests
        self.assertLessEqual(self.cumulative_freq_list[-1].cumulative, 1)
        for i in range(len(self.cumulative_freq_list) - 1, -1):
            self.assertEqual(
                self.cumulative_freq_list[i].cumulative,
                sum(self.cumulative_freq_list[j].cumulative for j in range(i)))

        # Test Histogram cumulatives

    def test_binary_search(self):
        for i, item in enumerate(self.cumulative_freq_list):
            with self.subTest(i=i):
                a = item.cumulative
                b = (self.cumulative_freq_list[i + 1].cumulative
                     if i < len(self.cumulative_freq_list) - 1 else 1)

                # Generate random number corresponding to current index so when
                # passed to Histogram.bin_search, we should get back i
                r = uniform(a, b)
                self.assertEqual(
                    Histogram.bin_search(self.cumulative_freq_list, r), i)

    def test_binary_search_edges(self):
        pass

    def test_custom_strip(self):
        IO = namedtuple("IO", "input expected")
        for i, io in enumerate([
                IO("", ""),
                IO("a", "a"),
                IO(" a", "a"),
                IO("4a", "4a"),
                IO("$3", "$3"),
                IO("£3 ", "£3"),
                IO("§∞§•¶•", ""),
                IO("fifty-five", "fifty-five"),
                IO("6%", "6%")
        ]):
            with self.subTest(i=i):
                self.assertEqual(Histogram.strip_non_alnums(io.input),
                                 io.expected)

    # TODO: Determine min number of samples for result to be statistically
    # significant
    # TODO: Determine DTW value that represents min similarity
    # TODO: Use dynamic time warping to measure similarity between expected and actual
    # TODO: See if divergence decreases with more iterations.

    def test_rand_word(self):
        pass

    def test_rand_word_edges(self):
        histogram = Histogram(word_to_freq=one_word_data.word_to_freq)

        # Randomly choosing one word from a list of one word should always
        # return that one word
        self.assertEqual(histogram.sample(), one_word_data.words[0])

    def test_uniform_histogram_matches_rand_word(self):

        expected_histogram = Histogram(
            word_to_freq=small_uniform_data.word_to_freq)
        n_samples = 500

        self.assertEqual(len(expected_histogram.word_to_freq),
                         len(small_uniform_data.word_to_freq))
        self.assertEqual(expected_histogram.tokens,
                         small_uniform_data.about.tokens)
        self.assertEqual(expected_histogram.types,
                         small_uniform_data.about.types)

        # make a tuple of words from distribution
        words = tuple(word for item in small_data.distribution
                      for word in item.words)

        # make a tuple of probabilities corresponding to words
        probs = tuple(
            Fraction(item.prob, len(item.words))
            for item in small_data.distribution for _ in item.words)

        # generate distribution modeling histogram.rand_word
        actual_wordfreq = generate_samples(n_samples,
                                           expected_histogram.sample)
        actual_histogram = Histogram(actual_wordfreq)

        self.assertLessEqual(expected_histogram.similarity(actual_histogram),
                             HistogramTestSuite.DIVERGENCE_THRESHOLD)

    @unittest.skip("leave this for later")
    def test_rand_word_samples_match_discrete_distribution(self):
        pass
