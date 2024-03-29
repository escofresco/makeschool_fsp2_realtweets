from collections import Counter, defaultdict, deque, Iterable
from contextlib import redirect_stdout
from functools import reduce, wraps
from io import StringIO
from math import ceil, floor, log2
from os import chmod, walk
from os.path import exists, join
from random import choice, randrange
from time import time

from .online import Avg

__all__ = [
    "LogMethodCalls", "sample_size", "is_binary_format", "map_to_binary",
    "invert_dict", "randints", "rand_word_distro", "binsearch",
    "capture_stdout", "recur_chmod", "p", "ismethod",
    "merge_nonsequentials_containing_ints", "merge_sequentials_containing_ints",
    "merge_data_containing_ints", "run_iter_code", "run_code"
]


class LogMethodCalls(type):
    """Metaclass creates instance variable called _logs_:deque, which holds
    most recent method calls of logs_size. _times_:defaultdict(online.Avg) holds
    the average times that methods take to execute.

    Follow Principle of Least Astonishment (POLA) by maintaining expected
    subclass behavior. Add _logs_ and _times_ features in a way that doesn't interfere
    with anticipated usability.

    adapted from: shorturl.at/ikW56
    """
    __slots__ = ()

    class Log:
        """Use this to manage a method's name and the number of times it's
        been called. sys.getsizeof shows this to be smaller than simple list."""
        __slots__ = ("name", "calls")

        def __init__(self, name, calls=1):
            self.name = name
            self.calls = calls

        def addcall(self):
            self.calls += 1

    def __new__(cls, name, bases, attrs, logs_size=2):
        for name, attr in attrs.items():
            if callable(attr):
                attrs[name] = LogMethodCalls.logdec(attr, logs_size)
            elif isinstance(attr, classmethod):
                attrs[name] = LogMethodCalls.logdec(attr.__func__, logs_size)

        if "__slots__" in attrs:
            # when the subclass is using slots, add in our internal
            # instance vars; cast to <set> to remove duplicates, then cast back
            # to tuple
            attrs["__slots__"] = tuple(
                set((*attrs["__slots__"], "_logs_", "_times_")))
        obj = type.__new__(cls, name, bases, attrs)
        obj.__new__ = LogMethodCalls.makenew(obj)
        return obj

    @staticmethod
    def logdec(method, logs_size):

        @wraps(method)
        def wrapper(self, *a, **ka):
            method_name = method.__name__
            if not len(self._logs_) or self._logs_[-1].name != method_name:
                self._logs_.append(LogMethodCalls.Log(method_name))
            else:
                self._logs_[-1].addcall()
            if len(self._logs_) > logs_size:
                self._logs_.popleft()
            time_0 = time()
            res = method(self, *a, **ka)
            duration = time() - time_0
            self._times_[method_name].add(duration)
            return res

        return wrapper

    @staticmethod
    def makenew(obj):

        def __new__(cls, *a, **ka):
            new_obj = super(obj, cls).__new__(cls)
            new_obj._logs_ = deque()
            new_obj._times_ = defaultdict(Avg)
            return new_obj

        return __new__


def sample_size(std_dev, margin_of_error, z_val=1.96):
    return ceil(((std_dev * z_val) / margin_of_error)**2)


def is_binary_format(string):
    try:
        int(string, 2)
    except ValueError:
        return False
    return True


def map_to_binary(array):
    if not len(array):
        return []
    bin_str_len = ceil(log2(len(array)))
    for i in range(len(array)):
        yield format(i, "0" + str(bin_str_len) + "b")


def invert_dict(d, merge_collisions=True):
    res = dict()
    for key, value in d.items():
        if isinstance(value, Iterable):
            for cur_value in value:
                if merge_collisions:
                    if value in res:
                        res[value].append(key)
                    else:
                        res[value] = [key]
                else:
                    res[value] = key
        else:
            if merge_collisions:
                if value in res:
                    res[value].append(key)
                else:
                    res[value] = [key]
            else:
                res[value] = key
    return res


def randints(n_ints,
             target_sum=None,
             min_val=None,
             max_val=None,
             deviation=None,
             variance=None):
    """Generate a list of n_ints integers (1 to n) that add up to target_sum.
    adapted from:
    http://sunny.today/generate-random-integers-with-fixed-sum/"""

    if not n_ints:
        # n_ints is zero or NaN
        return []

    target_sum = n_ints if target_sum is None else target_sum

    if target_sum < n_ints:
        raise ValueError("target_sum must at least be equal to number of ints")

    if deviation is not None and variance is not None:
        raise ValueError("A value can only be set for deviation or variance")
    mean = target_sum // n_ints  # guaranteed to be at least 1 # 1

    deviation = 1. if deviation is None else deviation

    # use variance to set bounds for random integer of each position
    variance = floor(deviation * mean) if variance is None else variance

    min_val = mean - variance if min_val is None else min_val
    max_val = mean + variance - 1 if max_val is None else max_val

    if min_val > max_val:
        min_val, max_val = max_val, min_val
    nums = [min_val] * n_ints  # initialize an array of integers
    remaining_sum = target_sum - min_val * n_ints  # the number of ones left to add
    while remaining_sum > 0:
        # as long as the sum of nums is less than target_sum,
        # add ones to random indices if the indeger at that index is less
        # than max_val.
        i = randrange(n_ints)
        if nums[i] <= max_val:
            nums[i] += 1
            remaining_sum -= 1
    return nums


def rand_word_distro(n_words, words, variance):
    pass


def binsearch(array, target):
    """Use binary search to find target in array.

    Args:
        array:  A sequence of data.
        target:
                If target is callable, it's expected to accept a comparison
                function that it uses to check it an element is a match.
                If target isn't callable, it just represents a value to
                find.
    """
    if not callable(target):
        ## Since target isn't callable, it is converted to something that is.
        ## It accepts a comparison function which takes a single argument to
        ## check against.
        target_val = target

        def is_match(cmp):
            res = cmp(target_val)
            if res is NotImplemented:
                return False
            return res

        target = is_match

    lo = 0
    hi = len(array) - 1
    while lo <= hi:
        mid = (hi + lo) // 2  # ?do we need to handle overflow in python?
        #if target(array[mid]):
        if target(array[mid].__eq__):
            return mid
        #if key(array[mid]) < target:
        if target(array[mid].__lt__):
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def capture_stdout(func, *args, **kwargs):
    """Capture the standard output of a function and return the string.

    Args:
        func: A function to be called.
        *args:
        **kwargs: The arguments that would be passed to func.

    pretty much copied from https://tinyurl.com/wkutfow
    """
    sio = StringIO()
    with redirect_stdout(sio):
        func(*args, **kwargs)
    return sio.getvalue()


def generate_samples(n_samples, func, *args, **kwargs):
    """Call a function a bunch of times and count the results.

    Args:
        n_samples: Number of time to call the function.
        func: The function results are counted from.
        *args
        **args: The arguments to pass to func.

    Returns:
        Counter containing results.
    """
    samples = Counter()
    for _ in range(n_samples):
        res = func(*args, **kwargs)
        samples[res] += 1
    return samples


def recur_chmod(dir, *modes):
    """Take a directory and variable number of arguments for mode (a file can
    have multiple permissions) and recursively set mode
    (octal, ex: stat.S_IEXEC) for files.

    Args:
        dir: Path for direction.
        *modes: The octal permission for a file.
    """
    if not exists(dir):
        raise NotADirectoryError(f"{dir} not found :^() ")

    # bitwise-or list of modes
    new_permisions = reduce(lambda x, y: x | y, modes)

    chmod(dir, new_permisions)
    for root, dirs, files in walk(dir):
        # adapted from https://stackoverflow.com/a/16265554/8011811
        for curdir in dirs:
            chmod(join(root, curdir), new_permisions)

        for curfile in files:
            chown(join(root, curfile), new_permisions)


def p(string):
    cols = 20
    print()
    print("~" * cols)
    print(string)
    print("~" * cols)


def ismethod(attr):
    return callable(attr) or isinstance(attr, (classmethod, staticmethod))


def merge_nonsequentials_containing_ints(*args):
    """takes dictionaries as arguments and merges them together by adding their
    values."""
    res = defaultdict(int)
    for arg in args:
        for k, v in arg.items():
            if not isinstance(v, (int, float, complex)):
                raise ValueError(f"All values must addable, not concatenable")
            res[k] += v
    return res


def merge_sequentials_containing_ints(*args):
    """takes a bunch of sequential data (of the same type and structured as
    <key,value>) and merges them together by adding (not concatenating) their
    values. data should be structured homogeneously.
    WARNING: duplicate items within an argument are also merged. strings not
    allowed."""
    if not len(args):
        return ()
    if isinstance(args[0], str):
        raise ValueError("Strings are't allowed.")
    res = {}
    for arg in args:
        for key, val in arg:
            res[key] = val + res.get(key, 0)
    return tuple(res.items())


def merge_data_containing_ints(*args):
    """takes iterable data (of the same type) and merges everything together
    so that values are added. arguments must be structured homogeneously."""
    if not len(args):
        return

    if isinstance(args[0], dict):
        return merge_nonsequentials_containing_ints(*args)
    return merge_sequentials_containing_ints(*args)


def run_iter_code(it, codesep, stdout_symbol):
    """

    """
    lines = []
    code_lines = []
    last_stdout = None
    codesep_is_open = False
    for line in it:
        if line == codesep:
            codesep_is_open ^= 1
        elif line == stdout_symbol:
            yield last_stdout  #lines.append(last_stdout)
        else:
            if codesep_is_open:
                last_stdout = capture_stdout(eval, line)
            yield line  #lines.append(line)
    if codesep_is_open:
        raise ValueError("Input is missing a closing tag.")
    #return "\n".join(lines)


def run_code(it, codesep="|><|", stdout_symbol="|<>|"):
    return "\n".join(run_iter_code(it, codesep, stdout_symbol))
