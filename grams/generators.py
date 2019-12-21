from collections import Counter, defaultdict, deque, namedtuple
from functools import reduce
import os

from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from .grams import Gram, Histogram
from .stats import FreqDist

__all__ = ["Markov"]


class Markov:
    """An implementation of a markov chain."""
    __slots__ = ("order", "memory", "model")

    # constants for signifying the beginning and end of a sentence
    START_TOKEN = "<|~~START~~|>"
    STOP_TOKEN = "<|~~STOP~~|>"

    def __init__(self, corpus, order=1):
        self.order = order
        self.memory = order + 1

        # use the corpus and a word order
        self.model = self._make_model(corpus)

    def _make_model(self, corpus):
        pass

    @staticmethod
    def pos_sents(block_text):
        """Convert multiline text into an array of sentences, with each word tagged
        with part of speech."""
        for sentence in sent_tokenize(block_text):
            yield pos_tag(word_tokenize(sentence))

    @staticmethod
    def print_nested_dict(d):
        stack = [(k, v, 1) for k, v in d.items()]
        while len(stack):
            key, value, tabs = stack.pop()

            if isinstance(value, dict):
                print("__" * tabs + str(key))
                stack.extend([(k, v, tabs + 1) for k, v in value.items()])
            else:
                print("__" * tabs + str(key) + ": " + str(value))


if __name__ == "__main__":
    from pprint import pprint
    txt = """Arfool is the best atrese I've ever tasted.
            Yea, I need to blash arfool.
            """

    recurd = lambda: defaultdict(recurd)
    transitions = defaultdict(Counter)
    word_pos_sents = tuple(Markov.pos_sents(txt))
    order = 1
    memory = order + 1
    last_words = deque(maxlen=memory)
    for sent in word_pos_sents:
        for word_pos in sent:
            last_words.append(word_pos)
            if len(last_words) == memory:
                transitions[last_words[0]][last_words[1]] += 1
                last_words.popleft()

    Markov.print_nested_dict(transitions)
    d = FreqDist(transitions)
