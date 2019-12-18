from collections import Counter, defaultdict, namedtuple, UserDict, UserList
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

    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ##
    #   Magic method overrides
    ##
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def test_len(self):
        distro = Distro([])
        self.assertEqual(0, len(distro))

        distro = Distro([1, 2, 3, 4])
        self.assertEqual(4, len(distro))

        distro = Distro({})
        self.assertEqual(0, len(distro))

        distro = Distro({'a': 1})
        self.assertEqual(1, len(distro))

        distro = Distro(())
        self.assertEqual(0, len(distro))

        distro = Distro((1, 2, 3, 4))
        self.assertEqual(4, len(distro))

    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ##
    #  Instance methods
    ##
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def test_find_in_dict(self):
        distro = Distro({"apple": "banana", "banana": "orange"})
        self.assertIsNone(distro.find("apple"))
        self.assertEqual(distro.find("banana"), "apple")
        self.assertEqual(distro.find("orange"), "banana")

    def test_find_in_multidim_dict(self):
        distro = Distro({
            "apple": ("banana", "bongo"),
            "banana": ("orange", "porange")
        })
        self.assertIsNone(distro.find("apple"))
        self.assertIsNone(distro.find("banana"))
        self.assertIsNone(distro.find("orange"))

        ## check if find works for whole elements
        self.assertEqual(distro.find(lambda cmp: cmp(("banana", "bongo"))),
                         "apple")
        self.assertEqual(distro.find(lambda cmp: cmp(("orange", "porange"))),
                         "banana")

        ## check if find works for partial elements
        def wrapper(target):

            def is_match(cmp):
                res = cmp(target)
                if res is NotImplemented:
                    return False
                return res

            return is_match

        bongo = wrapper(("banana", "bongo"))
        self.assertEqual(distro.find(bongo), "apple")

        self.assertEqual(distro.find(lambda cmp: cmp(("orange", "porange"))),
                         "banana")

    def test_find_in_multidim_str_tuple(self):
        str_tup_dist = Distro((
            ("A pear", ("has quite the lair.", 23)),
            ("But Mike's", ("would make a bear.", 89)),
            ("Once rocks", ("become super rare.", 1)),
        ))

        def wrapper(target):

            def is_match(cmp):
                res = cmp(target)
                if res is NotImplemented:
                    return False
                return res

            return is_match

        ### test string data
        ## first index
        phrase_freq = wrapper("A pear")
        self.assertEqual(str_tup_dist.find(phrase_freq), 0)
        self.assertEqual(str_tup_dist.find("A pear"), 0)

        phrase_freq = wrapper(("has quite the lair.", 23))
        self.assertEqual(str_tup_dist.find(phrase_freq), 0)
        self.assertEqual(str_tup_dist.find(("has quite the lair.", 23)), 0)

        # custom key
        phrase_freq = wrapper("has quite the lair.")
        self.assertEqual(
            str_tup_dist.find(phrase_freq, key=lambda elm: elm[0].__eq__), 0)

        phrase_freq = wrapper(23)
        self.assertEqual(
            str_tup_dist.find(phrase_freq, key=lambda elm: elm[1].__eq__), 0)

        ## second index
        phrase_freq = wrapper("But Mike's")
        self.assertEqual(str_tup_dist.find(phrase_freq), 1)
        self.assertEqual(str_tup_dist.find("But Mike's"), 1)

        phrase_freq = wrapper(("has quite the lair.", 23))
        self.assertEqual(str_tup_dist.find(phrase_freq), 0)
        self.assertEqual(str_tup_dist.find(("has quite the lair.", 23)), 0)

        ## third index
        phrase_freq = wrapper("Once rocks")
        self.assertEqual(str_tup_dist.find(phrase_freq), 2)
        self.assertEqual(str_tup_dist.find("Once rocks"), 2)

        phrase_freq = wrapper(("become super rare.", 1))
        self.assertEqual(str_tup_dist.find(phrase_freq), 2)
        self.assertEqual(
            str_tup_dist.find("become super rare.",
                              key=lambda elm: elm[0].__eq__), 2)

    def test_find_in_deep_multidim_int_tuple(self):
        int_tup_dist = Distro((
            (54, (("has quite the lair.", 23, ("NB",)), ("too long to read",
                                                         243, ("NN",)))),
            (876, (("would make a bear.", 89, ("TR",)), ("big and tall", 98347,
                                                         ("VBZ",)))),
            (435, (("become super rare.", 1, ("NN",)), ("and very hard to find",
                                                       0, ("NB",)))),
        ))

        def wrapper(target):

            def is_match(cmp):
                res = cmp(target)
                if res is NotImplemented:
                    return False
                return res

            return is_match

        ### test int data
        ## test first index
        phrase_freq = wrapper(54)
        self.assertEqual(int_tup_dist.find(phrase_freq), 0)

        # test int data, key casts to different target value
        phrase_freq = wrapper("54")
        self.assertEqual(
            int_tup_dist.find(phrase_freq, key=lambda elm: str(elm).__eq__), 0)

        phrase_freq = wrapper((("has quite the lair.", 23, ("NB",)),
                               ("too long to read", 243, ("NN"))))
        self.assertEqual(int_tup_dist.find(phrase_freq), 0)

        phrase_freq = wrapper(("has quite the lair.", 23, ("NB",)))
        self.assertEqual(
            int_tup_dist.find(phrase_freq,
                              key=(lambda elm: elm[0].__eq__
                                   if isinstance(elm, tuple) else elm.__eq__)),
            0)

        def key(elm):
            print(elm)
            if (isinstance(elm, tuple) and len(elm) and
                    isinstance(elm[0], tuple) and len(elm[0]) > 2 and
                    isinstance(elm[0][2], tuple) and len(elm[0][2])):
                print(elm)
                return elm[0][2][0].__eq__
            return elm.__eq__

        phrase_freq = wrapper("NB")
        self.assertEqual(int_tup_dist.find(phrase_freq, key=key), 0)

        ## second index
        phrase_freq = wrapper(876)
        self.assertEqual(int_tup_dist.find(phrase_freq), 1)

        # test int data, key casts to different target value
        phrase_freq = wrapper("876")
        self.assertEqual(
            int_tup_dist.find(phrase_freq, key=lambda elm: str(elm).__eq__), 1)



        ## third index
        phrase_freq = wrapper(435)
        self.assertEqual(int_tup_dist.find(phrase_freq), 2)

        # test int data, key casts to different target value
        phrase_freq = wrapper("435")
        self.assertEqual(
            int_tup_dist.find(phrase_freq, key=lambda elm: str(elm).__eq__), 2)

    def test_find_in_dict_subclass(self):
        distro = Distro(Counter({"apple": "banana", "banana": "orange"}))

        self.assertTrue(issubclass(type(distro.bins), dict))
        self.assertIsNone(distro.find("apple"))
        self.assertEqual(distro.find("banana"), "apple")
        self.assertEqual(distro.find("orange"), "banana")

        distro = Distro(UserDict({"apple": "banana", "banana": "orange"}).data)

        self.assertTrue(issubclass(type(distro.bins), dict))
        self.assertIsNone(distro.find("apple"))
        self.assertEqual(distro.find("banana"), "apple")
        self.assertEqual(distro.find("orange"), "banana")

    def test_find_edges(self):
        distro = Distro([])

        self.assertIsNone(distro.find("a"))
        self.assertIsNone(distro.find(lambda elm: elm == "a"))
        self.assertIsNone(distro.find(lambda elm: elm == "a", is_sorted=True))

        distro = Distro({})
        self.assertIsNone(distro.find("a"))
        self.assertIsNone(distro.find(lambda elm: elm == "a"))
        self.assertIsNone(distro.find(lambda elm: elm == "a", is_sorted=True))

        distro = Distro(Counter())
        self.assertIsNone(distro.find("a"))
        self.assertIsNone(distro.find(lambda elm: elm == "a"))
        self.assertIsNone(distro.find(lambda elm: elm == "a", is_sorted=True))

        distro = Distro(())
        self.assertIsNone(distro.find("a"))
        self.assertIsNone(distro.find(lambda elm: elm == "a"))
        self.assertIsNone(distro.find(lambda elm: elm == "a", is_sorted=True))


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
        self.assertEqual(freqdist.max_type_len,
                         max(len("Ms. Markov"), len("I'm an orange")))
        self.assertEqual(freqdist.min_type_len,
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
        with self.assertRaises(InvalidDataTypeError):
            FreqDist.cast(123, [1, 2, 3])

        # check with invalid target and valid default argument
        self.assertEqual(FreqDist.cast(123, [1, 2, 3], "issa string"), "123")

        # check with invalid target and invalid default argument
        with self.assertRaises(InvalidDataTypeError):
            FreqDist.cast(123, [1, 2, 3], (4, 5, 6))

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


class SampleTestSuite(unittest.TestCase):

    def test_init(self):
        uniform_freqdist = FreqDist([("apple", 2), ("banana", 2),
                                     ("orange", 2)])
        sample = Sample(uniform_freqdist)
        self.assertIs(sample.distribution, uniform_freqdist)
        self.assertEqual(sample.n, 3)

    def test_preprocessing(self):
        N_SAMPLES = 10000
        freqdist = FreqDist(
            (("orange", 10), ("yellow", 16), ("green", 8), ("blue", 20),
             ("grey", 8), ("indigo", 8), ("pink", 10)))
        sample = Sample(freqdist)

        self.assertEqual(sample.n, 7)
        # expected_alias = ("yellow", "", "blue", "yellow", "blue", "blue",
        #                   "blue")
        expected_alias = (1, None, 3, 1, 3, 3, 3)
        expected_prob = (Fraction(7, 8), Fraction(1, 1), Fraction(7, 10),
                         Fraction(29, 40), Fraction(7, 10), Fraction(7, 10),
                         Fraction(7, 8))

        sample.alias = expected_alias
        sample.prob = expected_prob
        #freqdist.show()

        # print(tuple((t,float(Fraction(f, 80))) for t,f in freqdist.bins))

        sample_counts = Counter()
        for _ in range(N_SAMPLES):
            sample_counts[sample.randword()] += 1

        samples = FreqDist(sample_counts)
        #samples.show()
