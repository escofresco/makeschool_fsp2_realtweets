from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from os import mkdir, rmdir
from os.path import join
import sys
import unittest

from grams.grams import Covergram, Gram, FreqDist
from grams.root_exceptions import *
from grams.utils import capture_stdout

class GramTestSuite(unittest.TestCase):

    def test_gram_correct_parent(self):
        self.assertEqual(Gram.__bases__, (FreqDist,))

    def test_similarity(self):
        first_distro = Gram((("apple", 1), ("day", 10000)))
        second_distro = Gram((("apple", 1), ("day", 10000)))
        expected = 0.  # distros are identical
        actual = first_distro.similarity(second_distro)
        self.assertEqual(expected, actual)

        first_distro = Gram((("apple", 1), ("day", 100000)))
        second_distro = Gram((("apple", 100000), ("day", 1)))
        expected = 1  # distros are very distance
        actual = first_distro.similarity(second_distro)
        self.assertAlmostEqual(expected, actual, 3)

    def test_show_edges(self):
        dgram = Gram({})
        tgram = Gram(())
        lgram = Gram([])

        with self.assertRaises(MissingDataError):
            dgram.show()

        with self.assertRaises(MissingDataError):
            tgram.show()

        with self.assertRaises(MissingDataError):
            lgram.show()

    def test_show(self):
        ## temporarily capture standard output and compare to an expected string
        # check that dictionaries, tuples, and lists are handled correctly
        dgram = Gram({"apple": 2})
        tgram = Gram((("apple", 2),))
        lgram = Gram([["apple", 2]])
        expected = "\napple: ▇▇ 2.00 \n\n"
        f = StringIO()
        with redirect_stdout(f):
            dgram.show()
            self.assertEqual(expected, f.getvalue())
            f.__init__()
            tgram.show()
            self.assertEqual(expected, f.getvalue())
            f.__init__()
            lgram.show()
            self.assertEqual(expected, f.getvalue())

        dgram = Gram({'one': 1, 'fish': 4, 'two': 1, 'red': 1, 'blue': 1})
        tgram = Gram(
            (('one', 1), ('fish', 4), ('two', 1), ('red', 1), ('blue', 1)))
        lgram = Gram([['one', 1], ['fish', 4], ['two', 1], ['red', 1],
                      ['blue', 1]])
        expected = "\nblue: ▇ 1.00 \nfish: ▇▇▇▇ 4.00 \none : ▇ 1.00 \nred : ▇ 1.00 \ntwo : ▇ 1.00 \n\n"

        self.assertCountEqual(
            Counter(expected.split(" ")),
            Counter(capture_stdout(dgram.show).split(" ")))
        self.assertEqual(expected, capture_stdout(tgram.show))
        self.assertEqual(expected, capture_stdout(lgram.show))
