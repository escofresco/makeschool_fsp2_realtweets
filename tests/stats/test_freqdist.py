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

from tests.data import (no_word_data, one_word_data, small_data,
                        small_uniform_data, DISTRO_DISTANCE_THRESHOLD)


@dataclass
class TD:
    color_tuple = (("orange", 10), ("yellow", 16), ("green", 8), ("blue", 20),
                   ("grey", 8), ("indigo", 8), ("pink", 10))


class FreqDistTestSuite(unittest.TestCase):

    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ##
    #   Initialization
    ##
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def test_init_dict_empty(self):
        freqdist = FreqDist({})

        self.assertEqual(freqdist.dtype, dict)
        self.assertEqual(freqdist.min_freq, float("inf"))
        self.assertEqual(freqdist.lowest_rank_types, None)
        self.assertEqual(freqdist.max_freq, float("-inf"))
        self.assertEqual(freqdist.highest_rank_types, None)
        self.assertEqual(freqdist.max_token_len, None)
        self.assertEqual(freqdist.min_token_len, None)
        self.assertEqual(freqdist.online_mean_freq, Avg())
        self.assertEqual(freqdist.online_freq_var, Var())
        self.assertEqual(freqdist.token_count, 0)
        self.assertEqual(freqdist.type_count, 0)
        self.assertEqual(freqdist.bins, {})
        self.assertEqual(freqdist.type_dtype, None)
        self.assertEqual(freqdist.bin_dtype, None)

    def test_init_list_empty(self):
        freqdist = FreqDist([])
        self.assertEqual(freqdist.dtype, list)
        self.assertEqual(freqdist.min_freq, float("inf"))
        self.assertEqual(freqdist.lowest_rank_types, None)
        self.assertEqual(freqdist.max_freq, float("-inf"))
        self.assertEqual(freqdist.highest_rank_types, None)
        self.assertEqual(freqdist.max_token_len, None)
        self.assertEqual(freqdist.min_token_len, None)
        self.assertEqual(freqdist.online_mean_freq, Avg())
        self.assertEqual(freqdist.online_freq_var, Var())
        self.assertEqual(freqdist.token_count, 0)
        self.assertEqual(freqdist.type_count, 0)
        self.assertEqual(freqdist.bins, [])
        self.assertEqual(freqdist.type_dtype, None)
        self.assertEqual(freqdist.bin_dtype, None)

    def test_init_tuple_empty(self):
        freqdist = FreqDist(())
        self.assertEqual(freqdist.dtype, tuple)
        self.assertEqual(freqdist.min_freq, float("inf"))
        self.assertEqual(freqdist.lowest_rank_types, None)
        self.assertEqual(freqdist.max_freq, float("-inf"))
        self.assertEqual(freqdist.highest_rank_types, None)
        self.assertEqual(freqdist.max_token_len, None)
        self.assertEqual(freqdist.min_token_len, None)
        self.assertEqual(freqdist.online_mean_freq, Avg())
        self.assertEqual(freqdist.online_freq_var, Var())
        self.assertEqual(freqdist.token_count, 0)
        self.assertEqual(freqdist.type_count, 0)
        self.assertEqual(freqdist.bins, ())
        self.assertEqual(freqdist.type_dtype, None)
        self.assertEqual(freqdist.bin_dtype, None)

    def test_init_dict_of_strings(self):

        # check string to frequency mapping
        tokens_freqs = {
            "Ms. Markov": 2,
            "I'm an orange": 90,
        }
        # manually create dict_items datatype; couldn't find in docs
        dict_items = type({}.items())

        freqdist = FreqDist(tokens_freqs)

        self.assertEqual(freqdist.dtype, dict)
        self.assertEqual(freqdist.min_freq, 2)
        self.assertTrue(
            np.array_equal(freqdist.lowest_rank_types, ["Ms. Markov"]))
        self.assertEqual(freqdist.max_freq, 90)
        self.assertTrue(
            np.array_equal(freqdist.highest_rank_types, ["I'm an orange"]))
        self.assertEqual(freqdist.max_token_len,
                         max(len("Ms. Markov"), len("I'm an orange")))
        self.assertEqual(freqdist.min_token_len,
                         min(len("Ms. Markov"), len("I'm an orange")))
        self.assertEqual(freqdist.online_mean_freq.avg, np.mean([2, 90]))
        self.assertEqual(freqdist.online_freq_var.var(), np.var([2, 90]))
        self.assertEqual(freqdist.token_count, 2 + 90)
        self.assertEqual(freqdist.type_count, 2)
        self.assertEqual(freqdist.bins, tokens_freqs)
        self.assertEqual(freqdist.type_dtype, str)
        self.assertEqual(freqdist.bin_dtype, dict_items)

    def test_init_counter(self):

        # check init works for dict subclasses
        tokens_freqs = Counter({
            "Ms. Markov": 2,
            "I'm an orange": 90,
        })
        # manually create dict_items datatype; couldn't find in docs
        dict_items = type({}.items())

        freqdist = FreqDist(tokens_freqs)

        # subclasses of dict are converted to dict
        self.assertEqual(freqdist.dtype, dict)
        self.assertEqual(freqdist.min_freq, 2)
        self.assertTrue(
            np.array_equal(freqdist.lowest_rank_types, ["Ms. Markov"]))
        self.assertEqual(freqdist.max_freq, 90)
        self.assertTrue(
            np.array_equal(freqdist.highest_rank_types, ["I'm an orange"]))
        self.assertEqual(freqdist.max_token_len,
                         max(len("Ms. Markov"), len("I'm an orange")))
        self.assertEqual(freqdist.min_token_len,
                         min(len("Ms. Markov"), len("I'm an orange")))
        self.assertEqual(freqdist.online_mean_freq.avg, np.mean([2, 90]))
        self.assertEqual(freqdist.online_freq_var.var(), np.var([2, 90]))
        self.assertEqual(freqdist.token_count, 2 + 90)
        self.assertEqual(freqdist.type_count, 2)
        self.assertCountEqual(freqdist.bins, tokens_freqs)
        self.assertEqual(freqdist.type_dtype, str)
        self.assertEqual(freqdist.bin_dtype, dict_items)

    def test_init_dict_of_tuples(self):

        # check string to frequency mapping
        tokens_freqs = {
            ("Ms. Markov",): 2,
            ("I'm an orange",): 90,
        }
        # manually create dict_items datatype; couldn't find in docs
        dict_items = type({}.items())

        freqdist = FreqDist(tokens_freqs)

        self.assertEqual(freqdist.dtype, dict)
        self.assertEqual(freqdist.min_freq, 2)
        self.assertTrue(
            np.array_equal(freqdist.lowest_rank_types, [("Ms. Markov",)]))
        self.assertEqual(freqdist.max_freq, 90)
        self.assertTrue(
            np.array_equal(freqdist.highest_rank_types, [("I'm an orange",)]))
        self.assertEqual(freqdist.max_token_len,
                         max(len(("Ms. Markov",)), len(("I'm an orange",))))
        self.assertEqual(freqdist.min_token_len,
                         min(len(("Ms. Markov",)), len(("I'm an orange",))))
        self.assertEqual(freqdist.online_mean_freq.avg, np.mean([2, 90]))
        self.assertEqual(freqdist.online_freq_var.var(), np.var([2, 90]))
        self.assertEqual(freqdist.token_count, 2 + 90)
        self.assertEqual(freqdist.type_count, 2)
        self.assertEqual(freqdist.bins, tokens_freqs)
        self.assertEqual(freqdist.type_dtype, tuple)
        self.assertEqual(freqdist.bin_dtype, dict_items)

    def test_init_no_sort_dict_of_tuples(self):

        # check string to frequency mapping
        tokens_freqs = {
            ("Ms. Markov",): 2,
            ("I'm an orange",): 90,
        }
        # manually create dict_items datatype; couldn't find in docs
        dict_items = type({}.items())

        freqdist = FreqDist(tokens_freqs, sort_data=False)

        self.assertEqual(freqdist.dtype, dict)
        self.assertEqual(freqdist.min_freq, 2)
        self.assertTrue(
            np.array_equal(freqdist.lowest_rank_types, [("Ms. Markov",)]))
        self.assertEqual(freqdist.max_freq, 90)
        self.assertTrue(
            np.array_equal(freqdist.highest_rank_types, [("I'm an orange",)]))
        self.assertEqual(freqdist.max_token_len,
                         max(len(("Ms. Markov",)), len(("I'm an orange",))))
        self.assertEqual(freqdist.min_token_len,
                         min(len(("Ms. Markov",)), len(("I'm an orange",))))
        self.assertEqual(freqdist.online_mean_freq.avg, np.mean([2, 90]))
        self.assertEqual(freqdist.online_freq_var.var(), np.var([2, 90]))
        self.assertEqual(freqdist.token_count, 2 + 90)
        self.assertEqual(freqdist.type_count, 2)
        self.assertEqual(freqdist.bins, tokens_freqs)
        self.assertEqual(freqdist.type_dtype, tuple)
        self.assertEqual(freqdist.bin_dtype, dict_items)

    def test_init_list_of_lists(self):

        # check string to frequency mapping
        tokens_freqs = [
            ["Ms. Markov", 2],
            ["I'm an orange", 90],
        ]
        freqdist = FreqDist(tokens_freqs)

        self.assertEqual(freqdist.dtype, list)
        self.assertEqual(freqdist.min_freq, 2)
        self.assertTrue(
            np.array_equal(freqdist.lowest_rank_types, ["Ms. Markov"]))
        self.assertEqual(freqdist.max_freq, 90)
        self.assertTrue(
            np.array_equal(freqdist.highest_rank_types, ["I'm an orange"]))
        self.assertEqual(freqdist.max_token_len,
                         max(len("Ms. Markov"), len("I'm an orange")))
        self.assertEqual(freqdist.min_token_len,
                         min(len("Ms. Markov"), len("I'm an orange")))
        self.assertEqual(freqdist.online_mean_freq.avg, np.mean([2, 90]))
        self.assertEqual(freqdist.online_freq_var.var(), np.var([2, 90]))
        self.assertEqual(freqdist.token_count, 2 + 90)
        self.assertEqual(freqdist.type_count, 2)
        self.assertEqual(freqdist.bins, sorted(tokens_freqs))
        self.assertEqual(freqdist.type_dtype, str)
        self.assertEqual(freqdist.bin_dtype, list)

    def test_init_no_sort_list_of_lists(self):

        # check string to frequency mapping
        tokens_freqs = [
            ["Ms. Markov", 2],
            ["I'm an orange", 90],
        ]
        freqdist = FreqDist(tokens_freqs, sort_data=False)

        self.assertEqual(freqdist.dtype, list)
        self.assertEqual(freqdist.min_freq, 2)
        self.assertTrue(
            np.array_equal(freqdist.lowest_rank_types, ["Ms. Markov"]))
        self.assertEqual(freqdist.max_freq, 90)
        self.assertTrue(
            np.array_equal(freqdist.highest_rank_types, ["I'm an orange"]))
        self.assertEqual(freqdist.max_token_len,
                         max(len("Ms. Markov"), len("I'm an orange")))
        self.assertEqual(freqdist.min_token_len,
                         min(len("Ms. Markov"), len("I'm an orange")))
        self.assertEqual(freqdist.online_mean_freq.avg, np.mean([2, 90]))
        self.assertEqual(freqdist.online_freq_var.var(), np.var([2, 90]))
        self.assertEqual(freqdist.token_count, 2 + 90)
        self.assertEqual(freqdist.type_count, 2)
        self.assertEqual(freqdist.bins, tokens_freqs)
        self.assertEqual(freqdist.type_dtype, str)
        self.assertEqual(freqdist.bin_dtype, list)

    def test_init_list_of_tuples(self):

        # check string to frequency mapping
        tokens_freqs = [
            ("Ms. Markov", 2),
            ("I'm an orange", 90),
        ]
        freqdist = FreqDist(tokens_freqs)

        self.assertEqual(freqdist.dtype, list)
        self.assertEqual(freqdist.min_freq, 2)
        self.assertTrue(
            np.array_equal(freqdist.lowest_rank_types, ["Ms. Markov"]))
        self.assertEqual(freqdist.max_freq, 90)
        self.assertTrue(
            np.array_equal(freqdist.highest_rank_types, ["I'm an orange"]))
        self.assertEqual(freqdist.max_token_len,
                         max(len("Ms. Markov"), len("I'm an orange")))
        self.assertEqual(freqdist.min_token_len,
                         min(len("Ms. Markov"), len("I'm an orange")))
        self.assertEqual(freqdist.online_mean_freq.avg, np.mean([2, 90]))
        self.assertEqual(freqdist.online_freq_var.var(), np.var([2, 90]))
        self.assertEqual(freqdist.token_count, 2 + 90)
        self.assertEqual(freqdist.type_count, 2)
        self.assertEqual(freqdist.bins, sorted(tokens_freqs))
        self.assertEqual(freqdist.type_dtype, str)
        self.assertEqual(freqdist.bin_dtype, tuple)

    def test_init_no_sort_list_of_tuples(self):

        # check string to frequency mapping
        tokens_freqs = [
            ("Ms. Markov", 2),
            ("I'm an orange", 90),
        ]
        freqdist = FreqDist(tokens_freqs, sort_data=False)

        self.assertEqual(freqdist.dtype, list)
        self.assertEqual(freqdist.min_freq, 2)
        self.assertTrue(
            np.array_equal(freqdist.lowest_rank_types, ["Ms. Markov"]))
        self.assertEqual(freqdist.max_freq, 90)
        self.assertTrue(
            np.array_equal(freqdist.highest_rank_types, ["I'm an orange"]))
        self.assertEqual(freqdist.max_token_len,
                         max(len("Ms. Markov"), len("I'm an orange")))
        self.assertEqual(freqdist.min_token_len,
                         min(len("Ms. Markov"), len("I'm an orange")))
        self.assertEqual(freqdist.online_mean_freq.avg, np.mean([2, 90]))
        self.assertEqual(freqdist.online_freq_var.var(), np.var([2, 90]))
        self.assertEqual(freqdist.token_count, 2 + 90)
        self.assertEqual(freqdist.type_count, 2)
        self.assertEqual(freqdist.bins, tokens_freqs)
        self.assertEqual(freqdist.type_dtype, str)
        self.assertEqual(freqdist.bin_dtype, tuple)

    def test_init_list_of_nestedtuples(self):

        # check string to frequency mapping
        tokens_freqs = [
            (("Ms. Markov",), 2),
            (("I'm an orange",), 90),
        ]
        freqdist = FreqDist(tokens_freqs)

        self.assertEqual(freqdist.dtype, list)
        self.assertEqual(freqdist.min_freq, 2)
        self.assertTrue(
            np.array_equal(freqdist.lowest_rank_types, [("Ms. Markov",)]))
        self.assertEqual(freqdist.max_freq, 90)
        self.assertTrue(
            np.array_equal(freqdist.highest_rank_types, [("I'm an orange",)]))
        self.assertEqual(freqdist.max_token_len,
                         max(len(("Ms. Markov",)), len(("I'm an orange",))))
        self.assertEqual(freqdist.min_token_len,
                         min(len(("Ms. Markov",)), len(("I'm an orange",))))
        self.assertEqual(freqdist.online_mean_freq.avg, np.mean([2, 90]))
        self.assertEqual(freqdist.online_freq_var.var(), np.var([2, 90]))
        self.assertEqual(freqdist.token_count, 2 + 90)
        self.assertEqual(freqdist.type_count, 2)
        self.assertEqual(freqdist.bins, sorted(tokens_freqs))
        self.assertEqual(freqdist.type_dtype, tuple)
        self.assertEqual(freqdist.bin_dtype, tuple)

    def test_init_no_sort_list_of_nestedtuples(self):

        # check string to frequency mapping
        tokens_freqs = [
            (("Ms. Markov",), 2),
            (("I'm an orange",), 90),
        ]
        freqdist = FreqDist(tokens_freqs, sort_data=False)

        self.assertEqual(freqdist.dtype, list)
        self.assertEqual(freqdist.min_freq, 2)
        self.assertTrue(
            np.array_equal(freqdist.lowest_rank_types, [("Ms. Markov",)]))
        self.assertEqual(freqdist.max_freq, 90)
        self.assertTrue(
            np.array_equal(freqdist.highest_rank_types, [("I'm an orange",)]))
        self.assertEqual(freqdist.max_token_len,
                         max(len(("Ms. Markov",)), len(("I'm an orange",))))
        self.assertEqual(freqdist.min_token_len,
                         min(len(("Ms. Markov",)), len(("I'm an orange",))))
        self.assertEqual(freqdist.online_mean_freq.avg, np.mean([2, 90]))
        self.assertEqual(freqdist.online_freq_var.var(), np.var([2, 90]))
        self.assertEqual(freqdist.token_count, 2 + 90)
        self.assertEqual(freqdist.type_count, 2)
        self.assertEqual(freqdist.bins, tokens_freqs)
        self.assertEqual(freqdist.type_dtype, tuple)
        self.assertEqual(freqdist.bin_dtype, tuple)

    def test_init_tuple_of_nestedtuples(self):

        # check string to frequency mapping
        tokens_freqs = (
            (("Ms. Markov",), 2),
            (("I'm an orange",), 90),
        )
        freqdist = FreqDist(tokens_freqs)

        self.assertEqual(freqdist.dtype, tuple)
        self.assertEqual(freqdist.min_freq, 2)
        self.assertTrue(
            np.array_equal(freqdist.lowest_rank_types, [("Ms. Markov",)]))
        self.assertEqual(freqdist.max_freq, 90)
        self.assertTrue(
            np.array_equal(freqdist.highest_rank_types, [("I'm an orange",)]))
        self.assertEqual(freqdist.max_token_len,
                         max(len(("Ms. Markov",)), len(("I'm an orange",))))
        self.assertEqual(freqdist.min_token_len,
                         min(len(("Ms. Markov",)), len(("I'm an orange",))))
        self.assertEqual(freqdist.online_mean_freq.avg, np.mean([2, 90]))
        self.assertEqual(freqdist.online_freq_var.var(), np.var([2, 90]))
        self.assertEqual(freqdist.token_count, 2 + 90)
        self.assertEqual(freqdist.type_count, 2)
        self.assertEqual(freqdist.bins, tuple(sorted(tokens_freqs)))
        self.assertEqual(freqdist.type_dtype, tuple)
        self.assertEqual(freqdist.bin_dtype, tuple)

    def test_init_no_sort_tuple_of_nestedtuples(self):

        # check string to frequency mapping
        tokens_freqs = (
            (("Ms. Markov",), 2),
            (("I'm an orange",), 90),
        )
        freqdist = FreqDist(tokens_freqs, sort_data=False)

        self.assertEqual(freqdist.dtype, tuple)
        self.assertEqual(freqdist.min_freq, 2)
        self.assertTrue(
            np.array_equal(freqdist.lowest_rank_types, [("Ms. Markov",)]))
        self.assertEqual(freqdist.max_freq, 90)
        self.assertTrue(
            np.array_equal(freqdist.highest_rank_types, [("I'm an orange",)]))
        self.assertEqual(freqdist.max_token_len,
                         max(len(("Ms. Markov",)), len(("I'm an orange",))))
        self.assertEqual(freqdist.min_token_len,
                         min(len(("Ms. Markov",)), len(("I'm an orange",))))
        self.assertEqual(freqdist.online_mean_freq.avg, np.mean([2, 90]))
        self.assertEqual(freqdist.online_freq_var.var(), np.var([2, 90]))
        self.assertEqual(freqdist.token_count, 2 + 90)
        self.assertEqual(freqdist.type_count, 2)
        self.assertEqual(freqdist.bins, tokens_freqs)
        self.assertEqual(freqdist.type_dtype, tuple)
        self.assertEqual(freqdist.bin_dtype, tuple)

    def test_init_invalid_datatype(self):
        with self.assertRaises(InvalidTypeError):
            FreqDist(123)

        with self.assertRaises(InvalidTypeError):
            FreqDist("123")

        with self.assertRaises(InvalidTypeError):
            FreqDist(set([1, 2, 3]))

        with self.assertRaises(InvalidTypeError):
            FreqDist(None)

    def test_make_table_from_invalid_input(self):
        """Pass data that shouldn't make it through the preprocessing step
        of initialization"""
        with self.assertRaises(HeterogeneousTypeError):
            tokens_freqs = (
                (("Ms. Markov",), 2),
                (["I'm an orange"], 90),
            )
            FreqDist(tokens_freqs)

        with self.assertRaises(InvalidTokenTypeError):
            tokens_freqs = (
                (["Ms. Markov"], 2),
                (["I'm an orange"], 90),
            )
            FreqDist(tokens_freqs)

        with self.assertRaises(InvalidFrequencyTypeError):
            tokens_freqs = (
                ("Ms. Markov", 2.),
                ("I'm an orange", 90),
            )
            FreqDist(tokens_freqs)

        with self.assertRaises(HeterogeneousTypeError):
            tokens_freqs = (
                (("Ms. Markov", "I think it's the perfect time for tea"), 2),
                (["I'm an orange"], 90),
            )
            FreqDist(tokens_freqs)

        with self.assertRaises(ImproperTupleFormatError):
            tokens_freqs = (
                (("Ms. Markov", "I think it's the perfect time for tea"), 2),
                (("I'm an orange",), 90),
            )
            FreqDist(tokens_freqs)

    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ##
    #   Magic method overrides
    ##
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def test_getitem(self):
        ## Nested tuple check
        tokens_freqs = (
            (("Ms. Markov",), 2),
            (("I'm an orange",), 90),
        )
        freqdist = FreqDist(tokens_freqs, sort_data=False)
        self.assertEqual(freqdist[0], (("Ms. Markov",), 2))

        freqdist = FreqDist(tokens_freqs)
        self.assertEqual(freqdist[0], (("I'm an orange",), 90))

        ## Simple dict check
        # check string to frequency mapping
        tokens_freqs = {
            "Ms. Markov": 2,
            "I'm an orange": 90,
        }
        freqdist = FreqDist(tokens_freqs)
        self.assertEqual(2, freqdist["Ms. Markov"])
        self.assertEqual(90, freqdist["I'm an orange"])

    def test_getitem_edges(self):
        ## Nested tuple check
        tokens_freqs = (
            (("Ms. Markov",), 2),
            (("I'm an orange",), 90),
        )
        freqdist = FreqDist(tokens_freqs, sort_data=False)
        with self.assertRaises(IndexNotFoundError):
            freqdist[2]

        with self.assertRaises(InvalidIndexError):
            freqdist[("Ms. Markov",)]

        ## Simple dict check
        # check string to frequency mapping
        tokens_freqs = {
            "Ms. Markov": 2,
            "I'm an orange": 90,
        }
        freqdist = FreqDist(tokens_freqs)
        with self.assertRaises(KeyNotFoundError):
            freqdist[[1, 2]]

        with self.assertRaises(KeyNotFoundError):
            freqdist["Mr. Markov"]

    def test_getitem_edges(self):
        tokens_freqs = (
            (("Ms. Markov",), 2),
            (("I'm an orange",), 90),
        )
        freqdist = FreqDist(tokens_freqs, sort_data=False)
        with self.assertRaises(IndexNotFoundError):
            freqdist[2]

    def test_len_represents_number_of_types(self):
        tokens_freqs = (
            (("Ms. Markov",), 2),
            (("I'm an orange",), 90),
        )
        freqdist = FreqDist(tokens_freqs, sort_data=False)
        self.assertEqual(2, len(freqdist))

        freqdist = FreqDist(tokens_freqs)
        self.assertEqual(2, len(freqdist))

    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ##
    #   Static methods
    ##
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def test_cast(self):
        # check with no default argument
        self.assertEqual(FreqDist.cast(123, "issa string"), "123")

        # check with invalid target
        with self.assertRaises(InvalidTypeError):
            FreqDist.cast(123, [1, 2, 3])

        # check with invalid target and valid default argument
        self.assertEqual(FreqDist.cast(123, [1, 2, 3], "issa string"), "123")

        # check with invalid target and invalid default argument
        with self.assertRaises(InvalidTypeError):
            FreqDist.cast(123, [1, 2, 3], (4, 5, 6))


    def test_jensen_shannon_distance(self):
        expected_bins = (("a", 0.), ("b", 1.))
        actual_bins = (*expected_bins,)
        self.assertEqual(
            FreqDist.jensen_shannon_distance(expected_bins, actual_bins), 0.)

        actual_bins = (("a", 1.), ("b", 0.))
        self.assertEqual(
            FreqDist.jensen_shannon_distance(expected_bins, actual_bins), 1.)

        # When binary words are passed directly, order shouldn't be preserved
        expected_bins = (("0", 0.), ("1", 1.))
        actual_bins = (*expected_bins,)
        self.assertEqual(
            FreqDist.jensen_shannon_distance(expected_bins, actual_bins), 0.)

        actual_bins = (("0", 1.), ("1", 0.))
        self.assertEqual(
            FreqDist.jensen_shannon_distance(expected_bins, actual_bins), 1.)

    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ##
    #  Instance methods
    ##
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def test_prob(self):
        tokens_freqs = (
            (("Ms. Markov",), 2),
            (("I'm an orange",), 90),
        )
        freqdist = FreqDist(tokens_freqs, sort_data=False)
        self.assertEqual(Fraction(90, 92), freqdist.prob(1))
        self.assertEqual(Fraction(2, 92), freqdist.prob(0))

        tokens_freqs = (("orange", 10), ("yellow", 16), ("green", 8),
                        ("blue", 20), ("grey", 8), ("indigo", 8), ("pink", 10))
        freqdist = FreqDist(tokens_freqs, sort_data=False)

        self.assertEqual(Fraction(1, 8), freqdist.prob(0))
        self.assertEqual(Fraction(1, 5), freqdist.prob(1))
        self.assertEqual(Fraction(1, 10), freqdist.prob(2))
        self.assertEqual(Fraction(1, 4), freqdist.prob(3))
        self.assertEqual(Fraction(1, 10), freqdist.prob(4))
        self.assertEqual(Fraction(1, 10), freqdist.prob(5))
        self.assertEqual(Fraction(1, 8), freqdist.prob(6))
