from collections import namedtuple
from fractions import Fraction
from math import floor
from random import randint, randrange
import unittest

import numpy as np

from grams.stats import FreqDist, Sample
from grams.utils import generate_samples, randints

from tests.data import (no_word_data, one_word_data, small_data,
                        small_uniform_data, DISTRO_DISTANCE_THRESHOLD)


class StatsTestSuite(unittest.TestCase):

    def assert_distro_and_data_equal(self, distro, data):
        for ivar in ("min", "max", "var", "mean", "std", "tokens", "types"):
            with self.subTest(instance_var=ivar):
                self.assertEqual(getattr(data.about, ivar),
                                 float(getattr(distro, ivar)))

        for expected, actual in zip(
                zip(*data.distribution),
            ("probs", "freqs", "cumulative_probs", "words")):
            with self.subTest(instance_var=actual):
                self.assertEqual(
                    expected,
                    (tuple(tuple(sorted(x)) for x in getattr(distro, actual))
                     if actual == "words" else getattr(distro, actual)))

    def test_empty_distro(self):
        distro = FreqDist(no_word_data.word_to_freq)
        self.assert_distro_and_data_equal(distro, no_word_data)

    def test_one_word_distro(self):
        distro = FreqDist(one_word_data.word_to_freq)
        self.assert_distro_and_data_equal(distro, one_word_data)

    def test_small_word_distro(self):
        distro = FreqDist(small_data.word_to_freq)
        self.assert_distro_and_data_equal(distro, small_data)

    def test_distro_handles_tuples(self):
        distro = FreqDist(
            tuple(
                (word, freq) for word, freq in small_data.word_to_freq.items()))
        self.assert_distro_and_data_equal(distro, small_data)

    def test_sample_null(self):
        sample = Sample(FreqDist(()))

        with self.assertRaises(ValueError):
            sample.rand()

    def test_sample_one_word(self):
        distro = FreqDist(one_word_data.word_to_freq)
        sample = Sample(distro)
        n_samples = 10000
        samples = generate_samples(n_samples, sample.randword)
        samples_distro = FreqDist(dict(samples))
        self.assertLessEqual(distro.similarity(samples_distro),
                             DISTRO_DISTANCE_THRESHOLD)

    def test_sample_n_words(self):
        distro = FreqDist(small_data.word_to_freq)
        sample = Sample(distro)
        self.assertEqual(len(tuple(sample.randwords(0))), 0)
        self.assertEqual(len(tuple(sample.randwords(1))), 1)
        self.assertEqual(len(tuple(sample.randwords(50))), 50)

    @unittest.skip("leave for later")
    def test_distro_similarity(self):

        for test_data in (no_word_data, one_word_data, small_data,
                          small_uniform_data):

            with self.subTest(data=test_data.words):
                first_distro = FreqDist(test_data.word_to_freq)
                second_distro = FreqDist(test_data.word_to_freq)
                third_distro = FreqDist(
                    tuple(zip(test_data.words, randints(len(test_data.words)))))

                # test the distance between distro and itself is zero
                self.assertIn(first_distro.similarity(second_distro),
                              {0., None})

                # test the commutative property holds
                self.assertEqual(first_distro.similarity(third_distro),
                                 third_distro.similarity(first_distro))

        first_distro = FreqDist(word_to_freq=(("apples", 100000), ("mangos",
                                                                   1)))
        second_distro = FreqDist((("apples", 1), ("mangos", 100000)))
        self.assertAlmostEqual(first_distro.similarity(second_distro), 1., 3)
        running_words = []
        running_words_set = set()

        for word in small_data.words:
            # go through a list of words, building up running_words, the
            # cumulative words up to each iteration. the average similarity
            # between a FreqDist(n random ints of sum k) and FreqDist(n 0s of sum 0)
            # should be 0.5.
            running_words.append(word)
            running_words_set.add(word)
            running_words_types = len(running_words_set)
            freqs = randints(running_words_types,
                             target_sum=randint(running_words_types,
                                                running_words_types + 1000))
            first_distro = FreqDist(
                word_to_freq=tuple(zip(running_words, freqs)))
            freqs = freqs[-1:] + freqs[:-1]  #

            second_distro = FreqDist(
                word_to_freq=tuple(zip(running_words, freqs)))
            with self.subTest(words=running_words, freqs=freqs):
                self.assertAlmostEqual(first_distro.similarity(second_distro),
                                       (1 if running_words_types > 1 else 0), 1)

    def test_distro_removedups(self):

        with self.assertRaises(ValueError):
            #FreqDist.is_valid_wordfreq("a", 0)
            tuple(FreqDist.remove_dups((("a", 0),)))

        with self.assertRaises(ValueError):
            tuple(FreqDist.remove_dups((("a", float('inf')),)))

        wordfreq = (("a", 1),)
        self.assertEqual(tuple(FreqDist.remove_dups(wordfreq)), wordfreq)
        wordfreq = (*wordfreq, *wordfreq)
        self.assertEqual(tuple(FreqDist.remove_dups(wordfreq)), (("a", 2),))

    @unittest.skip("asdf")
    def test_sample_small_uniform(self):
        distro = FreqDist(small_uniform_data.word_to_freq)
        sample = Sample(distro)
        n_samples = 10000
        samples = generate_samples(n_samples, distro, sample.rand)

    def test_get(self):
        distro = FreqDist(small_data.word_to_freq)

        # check a word that exists
        self.assertEqual(distro.get("the"), 3)

        # check non-existent word defaults to None
        self.assertEqual(distro.get("asdf"), None)

        # check non-existence word default to explicit val
        self.assertEqual(distro.get("asdf", 0), 0)
        self.assertEqual(distro.get("asdf", "Mumble"), "Mumble")

    def test_repr(self):
        distro = FreqDist([('one', 1), ('fish', 4), ('two', 1), ('red', 1),
                           ('blue', 1)])
        self.assertEqual(
            repr(distro),
            "word_to_freq: (('blue', 1), ('fish', 4), ('one', 1), ('red', 1), ('two', 1))\nmin: 1\nmax: 4\nmax_word_len: 4\nmean: 5/2\nvar: 9/4\nstd: 1.5\ntokens: 8\ntypes: 5\nwords: (('blue', 'one', 'red', 'two'), ('fish',))\nprobs: (Fraction(1, 2), Fraction(1, 2))\nfreqs: (1, 4)\ncumulative_probs: (Fraction(1, 2), Fraction(1, 1))"
        )
