from array import array
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial
from random import randrange
from typing import Iterable

import numpy as np


@dataclass
class Var:
    """Use Welford's online algorithm to track variance across a stream of
    data."""
    mean = count = sum = 0

    def add(self, val):
        self.count += 1
        delta = self.mean + Fraction(val - self.mean, self.count)
        self.sum += (val - self.mean) * (val - delta)
        self.mean = delta

    def remove(self, val):
        """Inverse of add
            self.mean + (val - self.mean) / self.count = delta
            self.mean * self.count + (val - self.mean) = delta * self.count
            self.mean * self.count - self.mean = delta * self.count - val
            self.mean * (self.count - 1) = delta * self.count - val
            self.mean = (delta * self.count - val) / (self.count - 1)
            delta = self.mean
            ∴ delta = (self.mean * self.count - val) / (self.count - 1)
        """
        if self.count <= 1:
            # Explicitly reset instance vars since alg is inexact
            self.mean = 0
            self.count = 0
            self.sum = 0
            return

        #delta = (self.mean * self.count - val) / (self.count - 1)
        #print("="*20)
        #print((self.mean * self.count - val), (self.count - 1))

        self.count -= 1
        delta = Fraction((self.mean * self.count - val), (self.count - 1))
        self.sum -= (val - self.mean) * (val - delta)
        self.mean = delta

    def std(self, ddof=0):
        return self.var(ddof=ddof)**.5

    def var(self, ddof=0):
        if not (self.count and self.sum):
            return 0
        return self.sum / (self.count - ddof)

    def __repr__(self):
        return f"mean: {self.mean} count: {self.count} sum: {self.sum}"


@dataclass
class Avg(float):
    """Track average across a stream of data."""
    avg: float = 0.
    total: int = 0

    def add(self, val):
        self.avg = (self.total * self.avg + val) / (self.total + 1)
        self.total += 1

    def remove(self, val):
        if not self.total:
            self.avg = 0.
            self.total = 0
        else:
            self.avg = (self.total * self.avg - val) / (self.total - 1)
            self.total = max(0, self.total - 1)

    def __float__(self):
        return self.avg

    def __int__(self):
        return int(round(self.avg))

    def __str__(self):
        return str(self.avg)


class Rand:
    """Use reservoir sampling to find random numbers across a stream of data."""
    __slots__ = ("sample_size", "cache_size", "cache")

    def __init__(self, it: Iterable, sample_size, max_repeats=3, key=None):

        self.sample_size = sample_size
        self.cache_size = max_repeats * self.sample_size
        self.cache = self._make_sample(it)

    def _make_sample(self, it, key):
        res = np.array([])
        for i, elm in enumerate(it):
            if key:
                key(
                    elm,
                    partial(Rand._sample_element,
                            i=i,
                            cache_size=cache_size,
                            array=res,
                            append_cb=np.append))
            else:
                Rand._sample_element(elm,
                                     i=i,
                                     cache_size=cache_size,
                                     array=res,
                                     append_cb=np.append)

        return res

    @staticmethod
    def _sample_element(elm, i, cache_size, array, append_cb):
        if i < cache_size:
            append_cb(res, elm)
        else:
            k = randrange(i + 1)
            if k < cache_size:
                array[k] = elm
