from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from os import mkdir, rmdir
from os.path import join
import sys
import unittest

from coverage import CoverageData

from grams.grams import Covergram, Gram, FreqDist
from grams.root_exceptions import *
from grams.utils import capture_stdout


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
        self.make_file(module_to_line[4][0], "")

        self.coverage_data.write_file(self.dummy_cov_filepath)

    def make_file(self, filepath, content):
        with open(filepath, "w+") as f:
            f.write(content)

    def test_super_init(self):
        #expected_module_to_coverage =
        pass
