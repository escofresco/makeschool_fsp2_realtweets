from collections import Counter, defaultdict
from dataclasses import dataclass
import pickle
import unittest

from grams.markov import Markov, MC, HMM
from grams.grams import Histogram
from grams.stats import FreqDist
from grams.utils import generate_samples


@dataclass
class TD:
    SIMPLE_CORPUS = "A man. A plan. A canal."
    SIMPLE_CORPUS_FIRSTORDER_TRANSITIONS = {
        ((MC.START_TOKEN, MC.START_TOKEN), (MC.START_TOKEN, MC.START_TOKEN)): {
            ((MC.START_TOKEN, MC.START_TOKEN), ("A", "DT")): 1,
        },
        ((MC.START_TOKEN, MC.START_TOKEN), ("A", "DT")): {
            (("A", "DT"), ("man", "NN")): 1,
            (("A", "DT"), ("plan", "NN")): 1,
            (("A", "DT"), ("canal", "NN")): 1,
        },
        (("A", "DT"), ("man", "NN")): {
            (("man", "NN"), (".", ".")): 1,
        },
        (("man", "NN"), (".", ".")): {
            ((".", "."), (MC.STOP_TOKEN, MC.STOP_TOKEN)): 1,
        },
        ((".", "."), (MC.STOP_TOKEN, MC.STOP_TOKEN)): {
            ((MC.STOP_TOKEN, MC.STOP_TOKEN), (MC.START_TOKEN, MC.START_TOKEN)):
                2,
        },
        ((MC.STOP_TOKEN, MC.STOP_TOKEN), (MC.START_TOKEN, MC.START_TOKEN)): {
            ((MC.START_TOKEN, MC.START_TOKEN), ("A", "DT")): 2,
        },
        (("A", "DT"), ("plan", "NN")): {
            (("plan", "NN"), (".", ".")): 1,
        },
        (("plan", "NN"), (".", ".")): {
            ((".", "."), (MC.STOP_TOKEN, MC.STOP_TOKEN)): 1,
        },
        (("A", "DT"), ("canal", "NN")): {
            (("canal", "NN"), (".", ".")): 1,
        },
        (("canal", "NN"), (".", ".")): {
            ((".", "."), (MC.STOP_TOKEN, MC.STOP_TOKEN)): 1,
        },
    }


class MCTestSuite(unittest.TestCase):
    maxDiff = None

    def test_startstop_tokens(self):
        self.assertEqual("<|~~START~~|>", MC.START_TOKEN)
        self.assertEqual("<|~~STOP~~|>", MC.STOP_TOKEN)

    def test_first_order_transitions(self):
        expected_transitions = TD.SIMPLE_CORPUS_FIRSTORDER_TRANSITIONS
        actual_transitions = MC("")._make_transitions(TD.SIMPLE_CORPUS)
        actual_transitions = {
            state: dict(transition)
            for state, transition in actual_transitions.items()
        }
        self.assertEqual(expected_transitions, actual_transitions)

    def test_first_order_generation(self):

        expected_sentences = set(
            self.combinations([
                "A man.", "A plan.", "A canal."
            ]))  # all possible combinations of the three sentences.
        markovchain = MC(TD.SIMPLE_CORPUS)
        expected_samples = FreqDist(
            tuple((sentence, 1) for sentence in expected_sentences))

        # Generate a bunch of three-sentence tuples
        actual_samples = FreqDist(
            generate_samples(
                10000, lambda: tuple(markovchain.generate_sentence()
                                     for _ in range(3))))

        for outcome, freq in actual_samples.bins.items():
            # confirm that each generated sentence is a possible combination
            self.assertIn(outcome, expected_sentences)

        self.assertLess(expected_samples.similarity(actual_samples), 0.05)

    # def test_save(self):
    #     def sattrs(obj):
    #         return {attr:getattr(obj, attr) for attr in obj.__slots__ if hasattr(obj, attr)}
    #     markovchain = MarkovChain(TD.SIMPLE_CORPUS)
    #     markovchain.save("test_chain.pkl")
    #     with open("test_chain.pkl", "rb") as f:
    #         _markovchain = pickle.load(f)
    #         for (attr, val), (_attr, _val) in zip(sorted(sattrs(markovchain).items()), sorted(sattrs(_markovchain).items())):
    #             with self.subTest(first_attrs=(attr,val), second_attrs=(_attr, _val)):
    #                 self.assertEqual(attr, _attr)
    #             with self.subTest(first_attrs=(attr,val), second_attrs=(_attr, _val)):
    #                 self.assertEqual(val, _val)

    def permutations(self, arr):

        def _permutations(i):
            if i == len(arr):
                res.append(tuple(arr))
            else:
                for j in range(i, len(arr)):
                    arr[i], arr[j] = arr[j], arr[i]
                    _permutations(i + 1)
                    arr[i], arr[j] = arr[j], arr[i]

        res = []
        _permutations(0)
        return res

    def combinations(self, arr):

        def _combinations(cur):
            if len(cur) == len(arr):
                res.append(tuple(cur))
            else:
                for elm in arr:
                    _combinations(cur + [elm])

        res = []
        _combinations([])
        return res
