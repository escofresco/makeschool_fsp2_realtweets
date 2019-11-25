from contextlib import redirect_stdout
from io import StringIO
from os import mkdir, rmdir
from os.path import join
import sys
import unittest

from coverage import CoverageData

from grams import Covergram, Gram, Distro


class GramTestSuite(unittest.TestCase):
    def test_gram_correct_parent(self):
        self.assertEqual(Gram.__bases__, (Distro, ))

    def test_line_as_words_edges(self):
        expected = ()
        line = ""
        self.assertEqual(expected, tuple(Gram.line_as_words(line)))

        expected = (["a"], )
        line = "a"
        self.assertEqual(expected, tuple(Gram.line_as_words(line)))

        line = "a."
        self.assertEqual(expected, tuple(Gram.line_as_words(line)))

        line = "a&*"
        self.assertEqual(expected, tuple(Gram.line_as_words(line)))

        line = "#@a"
        self.assertEqual(expected, tuple(Gram.line_as_words(line)))

        # test edge cases where a character is prefixed by dollar sign or euro.
        expected = (["$", "2"], ["b", "i", "l", "l"])
        line = "$2 bill!"
        self.assertEqual(expected, tuple(Gram.line_as_words(line)))

        expected = (["£", "2"], ["b", "i", "l", "l"])
        line = "£2 bill!"
        self.assertEqual(expected, tuple(Gram.line_as_words(line)))

        # test edge case where character is followed by percent sign
        expected = (["2", "%"], ["m", "i", "l", "k"])
        line = "2% milk:)"
        self.assertEqual(expected, tuple(Gram.line_as_words(line)))

    def test_line_as_words(self):
        expected = (["A", "n"], ["a", "p", "p", "l",
                                 "e"], ["a"], ["d", "a", "y"], ["i", "s"],
                    ["w", "h", "a", "t"], ["p", "e", "o", "p", "l",
                                           "e"], ["s", "a", "y"])
        line = "An apple a day is what people say."
        self.assertEqual(expected, tuple(Gram.line_as_words(line)))

    def test_similarity(self):
        first_distro = Distro((("apple", 1), ("day", 10000)))
        second_distro = Distro((("apple", 1), ("day", 10000)))
        expected = 0.  # distros are identical
        actual = first_distro.similarity(second_distro)
        self.assertEqual(expected, actual)

        first_distro = Distro((("apple", 1), ("day", 100000)))
        second_distro = Distro((("apple", 100000), ("day", 1)))
        expected = 1  # distros are very distance
        actual = first_distro.similarity(second_distro)
        self.assertAlmostEqual(expected, actual, 3)

    def test_visualizer_edges(self):
        dgram = Gram({})
        tgram = Gram(())
        lgram = Gram([])

        with self.assertRaises(ValueError):
            dgram.visualize()

        with self.assertRaises(ValueError):
            tgram.visualize()

        with self.assertRaises(ValueError):
            lgram.visualize()

    def test_visualizer(self):
        ## temporarily capture standard output and compare to an expected string
        # check that dictionaries, tuples, and lists are handled correctly
        dgram = Gram({"apple": 2})
        tgram = Gram((("apple", 2), ))
        lgram = Gram([["apple", 2]])
        expected = "\napple: ▇▇ 2.00 \n\n"
        f = StringIO()
        with redirect_stdout(f):
            dgram.visualize()
            self.assertEqual(expected, f.getvalue())
            f.__init__()
            tgram.visualize()
            self.assertEqual(expected, f.getvalue())
            f.__init__()
            lgram.visualize()
            self.assertEqual(expected, f.getvalue())

        dgram = Gram({'one': 1, 'fish': 4, 'two': 1, 'red': 1, 'blue': 1})
        tgram = Gram(
            (('one', 1), ('fish', 4), ('two', 1), ('red', 1), ('blue', 1)))
        lgram = Gram([['one', 1], ['fish', 4], ['two', 1], ['red', 1],
                      ['blue', 1]])
        expected = "\nblue: ▇ 1.00 \nfish: ▇▇▇▇ 4.00 \none : ▇ 1.00 \nred : ▇ 1.00 \ntwo : ▇ 1.00 \n\n"
        f = StringIO()
        with redirect_stdout(f):
            # dgram.visualize()
            # self.assertEqual(expected, f.getvalue())
            # f.__init__()
            tgram.visualize()
            self.assertEqual(expected, f.getvalue())
            f.__init__()
            lgram.visualize()
            self.assertEqual(expected, f.getvalue())

    def test_not_implemented(self):

        # subclass Gram and check that NotImplementedError is enforced
        c = type("C", (Gram, ), {})([])

        with self.assertRaises(NotImplementedError):
            c.frequency("apple")

        with self.assertRaises(NotImplementedError):
            c.sample()


class CovergramTestSuite(unittest.TestCase):
    def __init__(self, *a, **ka):
        super().__init__(*a, **ka)

        self.tempdir = "./temp/"
        #mkdir(self.tempdir)

        # make dummy .coverage file
        self.dummy_cov_filepath = join(self.tempdir, ".coverage")
        #self.coverage_data = self.make_coveragedata_from_file(self.dummy_cov_filepath)


    # def __del__(self):
    #     rmdir(self.tempdir)

    def make_coveragedata_from_file(self, filepath):

        module_to_line = (
            (join(self.tempdir, "module1.py"), [1, 3, 5, 7, 9]),
            (join(self.tempdir, "module2.py"), [2, 4, 6, 8, 10]),
            (join(self.tempdir, "module3.py"), [2]),
            (join(self.tempdir, "module4.py"), [1, 2, 3, 4, 5, 6, 7, 8, 9]),
            (join(self.tempdir, "module5.py"), []),
        )

        ### coverage needs actual module files, so generate them
        self.coverage_data = CoverageData()
        self.coverage_data.add_lines(dict(((module_to_line[0]),)))
        self.make_file(
            module_to_line[0][0], """
                       x = 1

                       y = 2

                       z = x+y

                       print(z)

                       assert z == 3
                       """)

        self.coverage_data.add_lines(dict(((module_to_line[1]),)))
        self.make_file(
            module_to_line[1][0], """

                       import sys

                       sys.path.join("asdf", "fdh.txt")

                       def foo(a,b):

                           return a-b

                       assert foo(10, 1) == 9

                       """)

        self.coverage_data.add_lines(dict(((module_to_line[2]),)))
        self.make_file(
            module_to_line[2][0], """

                       assert True is True
                       """)
        self.coverage_data.add_lines(dict(((module_to_line[3]),)))
        self.make_file(
            module_to_line[3][0], """
                        a = b = c = 3
                        l = ll = lll = []
                        l.append("asdf")
                        assert l is ll is lll
                        assert l[0] == "asdf"
                        assert len(l) == len(ll) == len(lll)
                        l.pop()
                        assert l is ll is lll
                       """)
        self.coverage_data.add_lines(dict(((module_to_line[4]),)))
        self.make_file(
            module_to_line[4][0], "")

        self.coverage_data.write_file(self.dummy_cov_filepath)

    def make_file(self, filepath, content):
        with open(filepath, "w+") as f:
            f.write(content)

    def test_super_init(self):
        #expected_module_to_coverage =
        pass
