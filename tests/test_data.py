from fractions import Fraction
import unittest

import numpy as np

from grams.utils import invert_dict
from data import one_word_data, small_data


class TestSuite(unittest.TestCase):
    def test_oneworddata(self):
        expected_words = ("word", )
        expected_word_count = 1
        expected_word_to_freq = {"word": 1}

        expected_probs = (Fraction(1, 1), )
        expected_freqs = (1, )
        expected_cumulative_probs = (Fraction(1, 1), )
        expected_d_words = (("word", ), )
        (actual_probs, actual_freqs, actual_cumulative_probs,
         actual_words) = zip(*tuple(map(tuple, one_word_data.distribution)))

        self.assertEqual(one_word_data.words, expected_words)
        self.assertEqual(one_word_data.about.tokens, expected_word_count)
        self.assertEqual(one_word_data.word_to_freq, expected_word_to_freq)

        self.assertEqual(actual_probs, expected_probs)
        self.assertEqual(actual_freqs, expected_freqs)
        self.assertEqual(actual_cumulative_probs, expected_cumulative_probs)
        self.assertEqual(actual_words, expected_d_words)

        self.assertEqual(one_word_data.about.min, 1)
        self.assertEqual(one_word_data.about.max, 1)
        self.assertEqual(one_word_data.about.mean, 1.)
        self.assertEqual(one_word_data.about.var, 0.)
        self.assertEqual(one_word_data.about.std, 0.)

    def test_smalldata(self):
        expected_words = ("the", "orange", "banana", "peach", "a", "or",
                          "good", "tasty", "banana", "or", "the", "or",
                          "tasty", "the", "orange")
        expected_word_count = 15
        expected_word_to_freq = {
            "the": 3,
            "orange": 2,
            "banana": 2,
            "peach": 1,
            "a": 1,
            "or": 3,
            "good": 1,
            "tasty": 2
        }
        freq_words = sorted(tuple(
            (k, tuple(sorted(v)))
            for k, v in invert_dict(expected_word_to_freq).items()),
                            key=lambda x: x[0])
        actual_distribution = tuple(map(tuple, small_data.distribution))
        (actual_probs, actual_freqs, actual_cumulative_probs,
         actual_words) = zip(*actual_distribution)
        expected_freqs, expected_d_words = zip(*freq_words)
        expected_probs = tuple(
            Fraction(freq * len(words), expected_word_count)
            for freq, words in freq_words)
        expected_cumulative_probs = tuple(
            np.cumsum(expected_probs, dtype="object"))

        self.assertEqual(small_data.words, expected_words)
        self.assertEqual(small_data.about.tokens, expected_word_count)
        self.assertEqual(dict(small_data.word_to_freq), expected_word_to_freq)
        self.assertEqual(small_data.about.types, 8)

        self.assertEqual(actual_probs, expected_probs)
        self.assertEqual(actual_freqs, expected_freqs)
        self.assertEqual(actual_cumulative_probs, expected_cumulative_probs)
        self.assertEqual(actual_words, expected_d_words)

        self.assertEqual(small_data.about.min, 1)
        self.assertEqual(small_data.about.max, 3)
        self.assertEqual(small_data.about.mean, np.mean(expected_freqs))
        self.assertEqual(small_data.about.var, np.var(expected_freqs))
        self.assertEqual(small_data.about.std, np.std(expected_freqs))
