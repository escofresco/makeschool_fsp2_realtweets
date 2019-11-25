from collections import Counter, defaultdict, deque, Iterable
from functools import wraps
from math import ceil, floor, log2
from random import choice, randrange

from dit.divergences import jensen_shannon_divergence
from dit import Distribution


class LogMethodCalls(type):
    """Metaclass creates instance variable called _logs_:deque, which holds
    most recent method calls of logs_size.

    Follow Principle of Least Astonishment (POLA) by maintaining expected
    subclass behavior. Add _logs_ feature in a way that doesn't interfere
    with anticipated usability. In other words, users can interact with
    subclass without knowing how this metaclass works.

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
            return method(self, *a, **ka)

        return wrapper

    @staticmethod
    def makenew(obj):
        def __new__(cls, *a, **ka):
            new_obj = super(obj, cls).__new__(cls)
            new_obj._logs_ = deque()
            return new_obj

        return __new__


def sample_size(std_dev, margin_of_error, z_val=1.96):
    return ceil(((std_dev * z_val) / margin_of_error)**2)


def histogram_similarity(first_distro_words, first_distro_probs,
                         second_distro_words, second_distro_probs):
    """
    Takes two distributions, each a tuple of words and their corresponding
    probabilities.
    """
    error_msg = lambda w, p: ValueError(
        f"the lengths of {w} and {p} must be equal")

    if len(first_distro_words) != len(first_distro_probs):
        raise error_msg(first_distro_words, first_distro_probs)

    if len(second_distro_words) != len(second_distro_probs):
        raise error_msg(second_distro_words, second_distro_probs)
    expected_distribution = Distribution(first_distro_words,
                                         first_distro_probs)
    actual_distribution = Distribution(second_distro_words,
                                       second_distro_probs)
    return jensen_shannon_divergence(
        [expected_distribution, actual_distribution])


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


def generate_samples(n, func):
    counts = Counter()
    for _ in range(n):
        res = func()
        counts[res] += 1
    return counts


def binsearch(array, target, key=lambda x: x):
    lo = 0
    hi = len(array) - 1
    while lo <= hi:
        mid = (hi + lo) // 2  # ?do we need to handle overflow in python?
        if key(array[mid]) == target:
            return mid
        if key(array[mid]) < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


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
