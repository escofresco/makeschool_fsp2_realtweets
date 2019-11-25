from random import randrange
import sys

import online

if __name__ == "__main__":
    with open("/usr/share/dict/words", "r") as word_file:
        print(' '.join(online.Rand(word_file, int(sys.argv[1]))))
