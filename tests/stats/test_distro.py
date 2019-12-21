from array import array
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


class TD:
    SORTED_COLOR_FISH_TUPLE = (("blue", 1), ("fish", 4), ("one", 1), ("red", 1),
                               ("two", 1))
    SHUFFLED_COLOR_FISH_TUPLE = (("fish", 4), ("one", 1), ("blue", 1),
                                 ("red", 1), ("two", 1))
    FISH_MAP = {"blue": 1, "fish": 4, "one": 1, "red": 1, "two": 1}


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
        with self.assertRaises(InvalidTypeError):
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

    def test_contains_sequence(self):
        distro = Distro(TD.SORTED_COLOR_FISH_TUPLE, is_sorted=True)
        self.assertIn("blue", distro)
        self.assertIn("fish", distro)
        self.assertIn("two", distro)

        distro = Distro(TD.SORTED_COLOR_FISH_TUPLE)
        self.assertIn("blue", distro)
        self.assertIn("fish", distro)
        self.assertIn("two", distro)

    def test_contains_mapping(self):
        distro = Distro(TD.FISH_MAP, is_sorted=True)
        self.assertIn("blue", distro)
        self.assertIn("fish", distro)
        self.assertIn("two", distro)

        distro = Distro(TD.FISH_MAP)
        self.assertIn("blue", distro)
        self.assertIn("fish", distro)
        self.assertIn("two", distro)

    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ##
    #  Instance methods
    ##
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def test_find_in_2d_tuple(self):
        distro = Distro(TD.SHUFFLED_COLOR_FISH_TUPLE)

        self.assertEqual(distro.find("fish"), 0)
        self.assertEqual(distro.find(4), 0)
        self.assertEqual(distro.find("one"), 1)
        self.assertEqual(distro.find(1), 1)
        self.assertEqual(distro.find("blue"), 2)
        self.assertEqual(distro.find("red"), 3)
        self.assertEqual(distro.find("two"), 4)

    def test_find_in_2d_sorted_tuple(self):
        distro = Distro(TD.SORTED_COLOR_FISH_TUPLE, is_sorted=True)

        self.assertEqual(distro.find("blue"), 0)
        self.assertEqual(distro.find(1), 0)
        self.assertEqual(distro.find("fish"), 1)
        self.assertEqual(distro.find(4), 1)
        self.assertEqual(distro.find("one"), 2)
        self.assertEqual(distro.find("red"), 3)
        self.assertEqual(distro.find("two"), 4)

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
                               ("too long to read", 243, ("NN",))))
        self.assertEqual(int_tup_dist.find(phrase_freq), 0)

        phrase_freq = wrapper(("has quite the lair.", 23, ("NB",)))
        self.assertEqual(
            int_tup_dist.find(phrase_freq,
                              key=(lambda elm: elm[0].__eq__
                                   if isinstance(elm, tuple) else elm.__eq__)),
            0)

        def key(elm):
            if (isinstance(elm, tuple) and len(elm) and
                    isinstance(elm[0], tuple) and len(elm[0]) > 2 and
                    isinstance(elm[0][2], tuple) and len(elm[0][2])):
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
        distro = Distro([], is_sorted=True)

        self.assertIsNone(distro.find("a"))
        self.assertIsNone(distro.find(lambda elm: elm == "a"))
        self.assertIsNone(distro.find(lambda elm: elm == "a"))

        distro = Distro({}, is_sorted=True)
        self.assertIsNone(distro.find("a"))
        self.assertIsNone(distro.find(lambda elm: elm == "a"))
        self.assertIsNone(distro.find(lambda elm: elm == "a"))

        distro = Distro(Counter(), is_sorted=True)
        self.assertIsNone(distro.find("a"))
        self.assertIsNone(distro.find(lambda elm: elm == "a"))
        self.assertIsNone(distro.find(lambda elm: elm == "a"))

        distro = Distro((), is_sorted=True)
        self.assertIsNone(distro.find("a"))
        self.assertIsNone(distro.find(lambda elm: elm == "a"))
        self.assertIsNone(distro.find(lambda elm: elm == "a"))

    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ##
    #   Static methods
    ##
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def test_validate_dict(self):
        dict_type = dict
        self.assertTrue(Distro.is_mapping(dict_type))
        self.assertEqual(Distro.classify_dtype(dict_type), dict)

    def test_validate_dict_subclasses(self):
        counter_type = type(Counter())
        self.assertTrue(Distro.is_mapping(counter_type))
        self.assertEqual(Distro.classify_dtype(counter_type), dict)

    def test_validate_list(self):
        list_type = list
        self.assertTrue(Distro.is_mutable_sequence(list_type))
        self.assertTrue(Distro.is_sequence(list_type))
        self.assertEqual(Distro.classify_dtype(list_type), list)

    def test_validate_tuple(self):
        tuple_type = tuple
        self.assertTrue(Distro.is_immutable_sequence(tuple_type))
        self.assertEqual(Distro.classify_dtype(tuple_type), tuple)

        tuple_type = type(TD.SORTED_COLOR_FISH_TUPLE)
        self.assertTrue(Distro.is_immutable_sequence(tuple_type))
        self.assertTrue(Distro.is_sequence(tuple_type))
        self.assertEqual(Distro.classify_dtype(tuple_type), tuple)
