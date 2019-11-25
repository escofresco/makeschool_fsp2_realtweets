# from concurrent.futures import ThreadPoolExecutor, TimeoutError
# from os import listdir
# from os.path import splitext
# from multiprocessing import cpu_count
# from sys import exit
# from unittest import main, TestLoader, TestResult, TestSuite, TextTestRunner
#
#
# max_workers = cpu_count()  # use cpu count as a the limit of processes
#
# # make a test load for each process
# test_suites = [TestSuite() for _ in range(max_workers)]
#
# test_dir = "tests"
#
# for i, filename in enumerate(listdir(test_dir)):
#     # go through files in test_dir by their filename, and could all test suites
#     # into new suites that fit into max_workers length
#     if filename.startswith("test_") and filename.endswith(".py"):
#         # filename corresponds to a test file
#         ## remove extension from filename
#         filename = splitext(filename)[0]  # https://tinyurl.com/tm6mr75
#
#         # append test_dir as a module
#         module_name = ".".join((test_dir, filename))
#
#         # load tests from suite given by dotted module string
#         loader = TestLoader().loadTestsFromName(module_name)
#
#         suite_idx = i % max_workers
#         test_suites[suite_idx].addTest(loader)
#
#
# def runner(test_suite):
#     test_result = TestResult()
#     test_suite.run(test_result)
#     if len(test_result.failures):
#         raise TimeoutError("Oh no! A test failed :O ")
#
#
# if __name__ == "__main__":
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = executor.map(runner, test_suites)
#         try:
#             for _ in futures:
#                 pass
#         except TimeoutError:
#             exit(1)
#     exit(0)
exit(1)
