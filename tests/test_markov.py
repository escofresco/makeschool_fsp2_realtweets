from collections import Counter, defaultdict
import unittest

from grams.markov import Markov


class MarkovTestSuite(unittest.TestCase):
    maxDiff = None
    TINY_CORPUS = "A man. A plan. A canal."
    START_TOKEN = "<|~~START~~|>"
    STOP_TOKEN = "<|~~STOP~~|>"

    def test_correctly_startstop_tokens(self):
        self.assertEqual(MarkovTestSuite.START_TOKEN, Markov.START_TOKEN)
        self.assertEqual(MarkovTestSuite.STOP_TOKEN, Markov.STOP_TOKEN)

    def test_word_map_is_correctly_generated_edges(self):
        corpus = ""
        markovmodel = Markov(corpus)
        expected = {}
        self.assertEqual(expected, markovmodel.model)

    def test_firstorder_word_map_is_correctly_generated(self):
        recursive_dict = lambda: defaultdict(recursive_dict)
        markovmodel = Markov(MarkovTestSuite.TINY_CORPUS)

        # build up solution manually from the timeless "A man. A plan. A canal."
        expected = recursive_dict()
        expected["A"]["DT"]["man"]["NN"] = 1
        expected["man"]["NN"]["."]["."] = 1
        expected["."]["."][MarkovTestSuite.STOP_TOKEN][
            MarkovTestSuite.STOP_TOKEN] = 3
        expected["A"]["DT"]["plan"]["NN"] = 1
        expected["plan"]["NN"]["."]["."] = 1
        expected["A"]["DT"]["canal"]["NN"] = 1
        expected["canal"]["NN"]["."]["."] = 1
        expected[MarkovTestSuite.START_TOKEN][
            MarkovTestSuite.START_TOKEN]["A"]["DT"] = 3
        expected[MarkovTestSuite.STOP_TOKEN][MarkovTestSuite.STOP_TOKEN][
            MarkovTestSuite.START_TOKEN][MarkovTestSuite.START_TOKEN] = 2

        self.assertEqual(expected, markovmodel.model)

    def test_secondorder_word_map_is_correctly_generated(self):
        recursive_dict = lambda: defaultdict(recursive_dict)
        markovmodel = Markov(MarkovTestSuite.TINY_CORPUS, order=2)

        # build up solution manually from the timeless "A man. A plan. A canal."
        expected = recursive_dict()

        # [START] A man
        expected[MarkovTestSuite.START_TOKEN][
            MarkovTestSuite.START_TOKEN]["A"]["DT"]["man"]["NN"] = 1

        # A man.
        expected["A"]["DT"]["man"]["NN"]["."]["."] = 1

        # man. [STOP]
        expected["man"]["NN"]["."]["."][MarkovTestSuite.STOP_TOKEN][
            MarkovTestSuite.STOP_TOKEN] = 1

        # . [STOP] [START]
        expected["."]["."][MarkovTestSuite.STOP_TOKEN][
            MarkovTestSuite.STOP_TOKEN][MarkovTestSuite.START_TOKEN][
                MarkovTestSuite.START_TOKEN] = 1

        # [STOP] [START] A
        expected[MarkovTestSuite.STOP_TOKEN][MarkovTestSuite.STOP_TOKEN][
            MarkovTestSuite.START_TOKEN][
                MarkovTestSuite.START_TOKEN]["A"]["DT"] = 1

        # [START] A plan
        expected[MarkovTestSuite.START_TOKEN][
            MarkovTestSuite.START_TOKEN]["A"]["DT"]["plan"]["NN"] = 1

        # A plan.
        expected["A"]["DT"]["plan"]["NN"]["."]["."] = 1

        # plan. [STOP]
        expected["plan"]["NN"]["."]["."][MarkovTestSuite.STOP_TOKEN][
            MarkovTestSuite.STOP_TOKEN] = 1

        # . [STOP] [START]
        expected["."]["."][MarkovTestSuite.STOP_TOKEN][
            MarkovTestSuite.STOP_TOKEN][MarkovTestSuite.START_TOKEN][
                MarkovTestSuite.START_TOKEN] += 1

        # [STOP] [START] A
        expected[MarkovTestSuite.STOP_TOKEN][MarkovTestSuite.STOP_TOKEN][
            MarkovTestSuite.START_TOKEN][
                MarkovTestSuite.START_TOKEN]["A"]["DT"] += 1

        # [START] A canal
        expected[MarkovTestSuite.START_TOKEN][
            MarkovTestSuite.START_TOKEN]["A"]["DT"]["canal"]["NN"] = 1

        # A canal.
        expected["A"]["DT"]["canal"]["NN"]["."]["."] = 1

        # canal. [STOP]
        expected["canal"]["NN"]["."]["."][MarkovTestSuite.STOP_TOKEN][
            MarkovTestSuite.STOP_TOKEN] = 1

        self.assertEqual(expected, markovmodel.model)

    def test_higherorder_word_map_is_correctly_generated(self):

        # from pprint import pprint
        # print("~"*20)
        # pprint(expected)
        # print("~"*20)
        # pprint(markovmodel.model)
        pass

    def test_sentences(self):
        expected = ((("A", "DT"), ("man", "NN"),
                     (".", ".")), (("A", "DT"), ("plan", "NN"), (".", ".")),
                    (("A", "DT"), ("canal", "NN"), (".", ".")))
        actual = tuple(Markov.sentences(MarkovTestSuite.TINY_CORPUS))
        self.assertEqual(expected, actual)

    def test_padded_sentence(self):
        expected = ((MarkovTestSuite.START_TOKEN, MarkovTestSuite.START_TOKEN),
                    ("A", "DT"), ("man", "NN"), (".", "."),
                    (MarkovTestSuite.STOP_TOKEN, MarkovTestSuite.STOP_TOKEN))
        actual = Markov.padded_sentence(
            (("A", "DT"), ("man", "NN"), (".", ".")))
        self.assertEqual(expected, actual)

    def test_get_nested_val_from_keys(self):
        expected = 123

        dictionary = {'a': {'b': {'c': {'d': 123}}}}

        ## test a value that exists
        keys = ['a', 'b', 'c', 'd']
        self.assertEqual(Markov.get_nested_val_from_keys(dictionary, keys),
                         expected)

        ## test a value that doesn't exist
        expected = None
        keys[0] = 'b'
        self.assertEqual(
            Markov.get_nested_val_from_keys(dictionary, keys, default=None),
            expected)

    def test_set_nested_val_from_keys(self):

        recursive_dict = lambda: defaultdict(recursive_dict)
        val = 1

        ## change a value for keys in an empty dictionary
        expected = recursive_dict()
        expected['a']['p']['p']['l']['e'] = val
        keys = ['a', 'p', 'p', 'l', 'e']
        self.assertEqual(
            Markov.set_nested_val_from_keys(recursive_dict(), keys, val),
            expected)

        ## change a value from an existing value
        dictionary = recursive_dict()
        dictionary['a']['p']['p']['l']['e'] = 2
        expected = recursive_dict()
        expected['a']['p']['p']['l']['e'] = val
        keys = ['a', 'p', 'p', 'l', 'e']

        # check dictionary contains the correct value
        self.assertEqual(dictionary['a']['p']['p']['l']['e'], 2)

        # check for the changed value
        self.assertEqual(
            Markov.set_nested_val_from_keys(dictionary, keys, val), expected)

        # check dictionary was edited in-place
        self.assertEqual(dictionary['a']['p']['p']['l']['e'], val)
