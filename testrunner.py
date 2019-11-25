from concurrent.futures import ThreadPoolExecutor, TimeoutError
from io import StringIO
from os import listdir
from os.path import splitext
from optparse import OptionParser
from multiprocessing import cpu_count
from sys import exit
from threading import get_ident
from unittest import main, TestLoader, TestResult, TestSuite, TextTestRunner

parser = OptionParser()
parser.add_option("-f",
                  "--failfast",
                  help="write report to FILE",
                  metavar="FILE")

# make a test load for each process
test_suites = []

test_dir = "tests"

for i, filename in enumerate(listdir(test_dir)):
    # go through files in test_dir by their filename, and could all test suites
    # into new suites that fit into max_workers length
    if filename.startswith("test_") and filename.endswith(".py"):
        # filename corresponds to a test file
        ## remove extension from filename
        filename = splitext(filename)[0]  # https://tinyurl.com/tm6mr75

        # append test_dir as a module
        module_name = ".".join((test_dir, filename))

        # load tests from suite given by dotted module string
        loader = TestLoader().loadTestsFromName(module_name)

        test_suites.append(loader)


def runner(test_suite):
    # test_result = TestResult()
    # test_suite.run(test_result)
    result_stream = StringIO()
    test_result = TextTestRunner(result_stream, failfast=True).run(test_suite)
    if len(test_result.failures):
        raise TimeoutError("Oh no! A test failed :O ")
    print("~" * 20)
    print(result_stream.getvalue())
    return result_stream.getvalue(), get_ident()


if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:
        futures = executor.map(runner, test_suites)
        try:
            for _ in futures:
                pass
        except TimeoutError:
            # signal failure
            exit(1)
        # for test_sui
        # try:
        #     test_result, thread_id = executor.submit(runner, test_suite)
    # signal success
    exit(0)
