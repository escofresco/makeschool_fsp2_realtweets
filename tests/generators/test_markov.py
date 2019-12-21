from collections import Counter, defaultdict
import unittest

from grams.generators import Markov


class MarkovTestSuite(unittest.TestCase):
    maxDiff = None

    def test_startstop_tokens(self):
        self.assertEqual("<|~~START~~|>", Markov.START_TOKEN)
        self.assertEqual("<|~~STOP~~|>", Markov.STOP_TOKEN)

    def test_first_order_transitions(self):
        pass
