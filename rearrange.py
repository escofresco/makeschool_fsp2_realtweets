from random import randrange
import sys


def fisher_yates(arr):
    for i in range(len(arr) - 1, 0, -1):
        randidx = randrange(i + 1)
        arr[i], arr[randidx] = arr[randidx], arr[i]

    return arr


if __name__ == "__main__":
    print(' '.join(fisher_yates(sys.argv[1:])))
