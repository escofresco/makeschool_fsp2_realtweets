from collections import Counter, defaultdict, deque
from functools import reduce
import os

from nltk import pos_tag, sent_tokenize, word_tokenize
from .grams import Gram


class Markov:
    """An implementation of a simple markov model with variable order."""
    __slots__ = ("order", "model")

    # constants for signifying the beginning and end of a sentence
    START_TOKEN = "<|~~START~~|>"
    STOP_TOKEN = "<|~~STOP~~|>"

    def __init__(self, corpus, order=1):
        self.order = order

        # use the corpus and a word order
        self.model = self._make_model(corpus)

    # def _make_model(self, corpus):
    #     res = defaultdict(Counter)
    #     for sentence in Markov.sentences(corpus):
    #         # go through each sentence
    #         last = None
    #         for token,pos in Markov.padded_sentence(sentence):
    #             # go through each (<token>, <part of speech>), updating res
    #             # with tokens following last, grouped by part of speech
    #             if last:
    #                 res[last[0]][last[1]][token][pos] += 1
    #             last = (token, pos)
    #     return res
    def _make_model(self, corpus):
        """A one-time generation step which builds up the model with rank equal
        to 2*self.order + 1
        {<token>: {
          <pos 1>: {
              <first token found after <token>>: {
                  <nth token found after <(n-1)th token after <token>>>:
                      <number of occurence for path to nth token from <token>
              }
          <pos 2>: {
              ...
          }
        <another token>: {
          ...
        }}
        """
        buffer_size = 2 * (self.order + 1)

        # recursively make defaultdict
        multidim_dict = lambda: defaultdict(multidim_dict)

        # this is the solution to self.model
        res = multidim_dict()

        # track self.order tokens following a token, which should contain
        # a token followed by part of speech, size being 2*(order+1)
        circular_buffer = deque(maxlen=buffer_size)
        for sentence in Markov.sentences(corpus):
            ## go through each sentence, building up res
            # pad sentence with start and stop tokens
            #   NOTICE: stop tokens are allowed to have sequences following
            #           them. this is useful information about sentence
            #           sequences.
            sentence = Markov.padded_sentence(sentence)

            for token_pos in sentence:
                circular_buffer.extend(token_pos)
                if len(circular_buffer) == buffer_size:
                    ## buffer is full, we can safely add sequence to the solution.
                    Markov.set_nested_val_from_keys(
                        res,
                        circular_buffer,
                        # increment current count of sequence (the path to
                        # count) by 1
                        Markov.get_nested_val_from_keys(res, circular_buffer) +
                        1)
        return res

    @staticmethod
    def sentences(corpus):
        for sentence in sent_tokenize(corpus):
            yield tuple(pos_tag(word_tokenize(sentence)))

    @staticmethod
    def padded_sentence(sentence):
        return ((Markov.START_TOKEN, Markov.START_TOKEN), *sentence,
                (Markov.STOP_TOKEN, Markov.STOP_TOKEN))

    @staticmethod
    def get_nested_val_from_keys(dictionary, keys, default=0):
        """Given a multidimensional dictionary, get the value for keys.
        ex: keys=('a', 'b', 'c', 'd')
            dictionary={
                'a': {
                    'b': {
                        'c': {
                            'd': 123
                        }
                    }
                }
            }
            returns 123
        """
        cur = dictionary
        for key in keys:
            if cur is None:
                return default
            cur = cur.get(key)
        return cur

    @staticmethod
    def set_nested_val_from_keys(dictionary, keys, val):
        """Given a *recursively defined* multidimensional dictionary, set the value `val` at `keys`. Edit is done in-place.
        ex: keys=('a', 'b', 'c', 'd')
            val=123
            dictionary={}

            returns {
                'a': {
                    'b': {
                        'c': {
                            'd': 123
                        }
                    }
                }
            }
        """
        subdict = dictionary
        for i in range(len(keys) - 1):
            subdict = subdict[keys[i]]
        subdict[keys[-1]] = val
        return dictionary
