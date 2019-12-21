from collections import namedtuple
from fractions import Fraction
from os.path import dirname, join
from random import uniform
import unittest

from data import one_word_data, small_data, small_uniform_data
from grams.grams import Histogram
from grams.utils import sample_size, map_to_binary


class HistogramTestSuite(unittest.TestCase):
    pass
