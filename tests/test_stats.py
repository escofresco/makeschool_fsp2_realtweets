from collections import namedtuple
from fractions import Fraction
from math import floor
from random import randint, randrange
import unittest

import numpy as np

from grams.online import Avg, Var
from grams.root_exceptions import *
from grams.stats import Distro, FreqDist, Sample
from grams.utils import generate_samples, randints

from tests.data import (no_word_data, one_word_data, small_data,
                        small_uniform_data, DISTRO_DISTANCE_THRESHOLD)


class DistroTestSuite(unittest.TestCase):

    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ##
    #   Initialization
    ##
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def test_init_dict_empty(self):
        distro = Distro({})

        self.assertEqual({}, distro.bins)
        self.assertEqual(dict, distro.dtype)
        self.assertEqual(None, distro.bin_dtype)

    def test_init_list_empty(self):
        distro = Distro([])
        self.assertEqual([], distro.bins)
        self.assertEqual(list, distro.dtype)
        self.assertEqual(None, distro.bin_dtype)

    def test_init_tuple_empty(self):
        distro = Distro(())
        self.assertEqual((), distro.bins)
        self.assertEqual(tuple, distro.dtype)
        self.assertEqual(None, distro.bin_dtype)

    def test_init_edges(self):
        with self.assertRaises(InvalidDataTypeError):
            distro = Distro("")


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
        self.assertEqual(freqdist.max_type_len, float("-inf"))
        self.assertEqual(freqdist.min_type_len, float("inf"))
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
        self.assertEqual(freqdist.max_type_len, float("-inf"))
        self.assertEqual(freqdist.min_type_len, float("inf"))
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
        self.assertEqual(freqdist.max_type_len, float("-inf"))
        self.assertEqual(freqdist.min_type_len, float("inf"))
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
        self.assertEqual(freqdist.max_type_len,
                         max(len("Ms. Markov"), len("I'm an orange")))
        self.assertEqual(freqdist.min_type_len,
                         min(len("Ms. Markov"), len("I'm an orange")))
        self.assertEqual(freqdist.online_mean_freq.avg, np.mean([2, 90]))
        self.assertEqual(freqdist.online_freq_var.var(), np.var([2, 90]))
        self.assertEqual(freqdist.token_count, 2 + 90)
        self.assertEqual(freqdist.type_count, 2)
        self.assertEqual(freqdist.bins, tokens_freqs)
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
        self.assertEqual(freqdist.max_type_len,
                         max(len(("Ms. Markov",)), len(("I'm an orange",))))
        self.assertEqual(freqdist.min_type_len,
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
        self.assertEqual(freqdist.max_type_len,
                         max(len(("Ms. Markov",)), len(("I'm an orange",))))
        self.assertEqual(freqdist.min_type_len,
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
        self.assertEqual(freqdist.max_type_len,
                         max(len("Ms. Markov"), len("I'm an orange")))
        self.assertEqual(freqdist.min_type_len,
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
        self.assertEqual(freqdist.max_type_len,
                         max(len("Ms. Markov"), len("I'm an orange")))
        self.assertEqual(freqdist.min_type_len,
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
        self.assertEqual(freqdist.max_type_len,
                         max(len("Ms. Markov"), len("I'm an orange")))
        self.assertEqual(freqdist.min_type_len,
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
        self.assertEqual(freqdist.max_type_len,
                         max(len("Ms. Markov"), len("I'm an orange")))
        self.assertEqual(freqdist.min_type_len,
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
        self.assertEqual(freqdist.max_type_len,
                         max(len(("Ms. Markov",)), len(("I'm an orange",))))
        self.assertEqual(freqdist.min_type_len,
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
        self.assertEqual(freqdist.max_type_len,
                         max(len(("Ms. Markov",)), len(("I'm an orange",))))
        self.assertEqual(freqdist.min_type_len,
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
        self.assertEqual(freqdist.max_type_len,
                         max(len(("Ms. Markov",)), len(("I'm an orange",))))
        self.assertEqual(freqdist.min_type_len,
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
        self.assertEqual(freqdist.max_type_len,
                         max(len(("Ms. Markov",)), len(("I'm an orange",))))
        self.assertEqual(freqdist.min_type_len,
                         min(len(("Ms. Markov",)), len(("I'm an orange",))))
        self.assertEqual(freqdist.online_mean_freq.avg, np.mean([2, 90]))
        self.assertEqual(freqdist.online_freq_var.var(), np.var([2, 90]))
        self.assertEqual(freqdist.token_count, 2 + 90)
        self.assertEqual(freqdist.type_count, 2)
        self.assertEqual(freqdist.bins, tokens_freqs)
        self.assertEqual(freqdist.type_dtype, tuple)
        self.assertEqual(freqdist.bin_dtype, tuple)


    def test_init_invalid_datatype(self):
        with self.assertRaises(InvalidDataTypeError):
            FreqDist(123)

        with self.assertRaises(InvalidDataTypeError):
            FreqDist("123")

        with self.assertRaises(InvalidDataTypeError):
            FreqDist(set([1, 2, 3]))

        with self.assertRaises(InvalidDataTypeError):
            FreqDist(None)

    def test_make_table_from_invalid_input(self):
        """Pass data that shouldn't make it through the preprocessing step
        of initialization"""
        with self.assertRaises(InvalidTokenDatatype):
            tokens_freqs = (
                (("Ms. Markov",), 2),
                ([
                    "I'm an orange",
                ], 90),
            )
            FreqDist(tokens_freqs)

        with self.assertRaises(InvalidTokenError):
            tokens_freqs = (
                ([
                    "Ms. Markov",
                ], 2),
                ([
                    "I'm an orange",
                ], 90),
            )
            FreqDist(tokens_freqs)

        with self.assertRaises(InvalidFrequencyError):
            tokens_freqs = (
                ("Ms. Markov", 2.),
                ("I'm an orange", 90),
            )
            FreqDist(tokens_freqs)

        with self.assertRaises(InvalidTokenDatatype):
            tokens_freqs = (
                (("Ms. Markov", "I think it's the perfect time for tea"), 2),
                ([
                    "I'm an orange",
                ], 90),
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
    #   Static methods
    ##
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def test_cast(self):
        # check with no default argument
        self.assertEqual(FreqDist.cast(123, "issa string"), "123")

        # check with invalid target
        with self.assertRaises(InvalidDataTypeError):
            FreqDist.cast(123, [1, 2, 3])

        # check with invalid target and valid default argument
        self.assertEqual(FreqDist.cast(123, [1, 2, 3], "issa string"), "123")

        # check with invalid target and invalid default argument
        with self.assertRaises(InvalidDataTypeError):
            FreqDist.cast(123, [1, 2, 3], (4, 5, 6))

class SampleTestSuite(unittest.TestCase):

    def test_
