from collections import Counter
import unittest

from grams.markov import Markov

class MarkovTestSuite(unittest.TestCase):
    def test_word_map_is_correctly_generated(self):
        corpus = ("A man.", "A plan.", "A canal.")
        markovmodel = Markov(corpus)
        expected = {"A": Counter(("man", "plan", "canal",)),
                    "man": Counter(("A",)),
                    "plan": Counter(("A",)),
                    "canal": Counter(("A",)),
                    }
        self.assertEqual(expected, markovmodel.word_map)

    def test_word_map_is_correctly_generated_edges(self):
        corpus = ("",)
        markovmodel = Markov(corpus)
        expected = {}
        self.assertEqual(expected, markovmodel.word_map)
