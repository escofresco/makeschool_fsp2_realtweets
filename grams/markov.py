from collections import Counter, defaultdict, deque, namedtuple
from functools import reduce
from itertools import chain
import os

from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from .grams import Gram, Histogram
from .stats import FreqDist
from .utils import capture_stdout

__all__ = ["MarkovChain"]


class MarkovChain:
    """An implementation of a markov chain."""
    __slots__ = ("order", "order", "memory", "use_pos", "start_token", "stop_token", "start_state", "chain")

    START_TOKEN = "<|~~START~~|>"
    STOP_TOKEN = "<|~~STOP~~|>"

    def __init__(self, corpus, order=1, use_pos=True):
        self.order = order
        self.memory = self.order + 2
        self.use_pos = use_pos

        # use start and stop to signify the beginning and end of a sentence.
        self.start_token = MarkovChain.START_TOKEN
        self.stop_token = MarkovChain.STOP_TOKEN
        if use_pos:
            self.start_token = (self.start_token, self.start_token)
            self.stop_token = (self.stop_token, self.stop_token)

        self.start_state = (self.start_token,)*(self.memory - 1)

        # use the corpus and a word order
        self.chain = dict(self._make_chain(self._make_transitions(corpus)))

    def __str__(self):
        res = []
        for state, transition_hist in self.chain.items():
            res.extend((f"\n{'~'*10} {state} {'~'*10}", capture_stdout(transition_hist.show)))
        return "\n".join(res)

    def _make_transitions(self, corpus):
        transitions = defaultdict(Counter)

        # Fill cache with start tokens. Since a start token is added at the
        # beginning of each iteration, don't add the last two.
        cache = deque(self.start_state[:-1], maxlen=self.memory)

        for sentence in (Gram.pos_sents(corpus) if self.use_pos else Gram.sents(corpus)):
            ## Depending on if pos_tags are being used, look through either
            ## (token, pos) pairs or just the tokens
            for word_pos in chain((self.start_token,), sentence, (self.stop_token,)):
                cache.append(word_pos)
                if len(cache) == self.memory:
                    ## cache is full
                    t_cache = tuple(cache)
                    state = t_cache[:-1]
                    transition = t_cache[1:]
                    transitions[state][transition] += 1
                    cache.popleft()
        return transitions

    def _make_chain(self, transitions):
        for state, transition in transitions.items():
            yield state, Histogram(tokens_freqs=transition)

    def _generate_sentence(self):
        last_state = self.start_state
        sentence = []
        while last_state[-1] != self.stop_token:
            ## as long as the last token doesn't equal the stop token, add the
            ## next token to sentence.
            # add only the first token of last_state
            last_token = last_state[0]
            if last_token != self.start_token:
                ## as long as last token isn't a start token, add it to
                ## sentence.
                sentence.append(last_token)
            last_state = self.next_state(last_state)

        # since the first index of last_state has been added to sentences at
        # this point, return the current sentence plus the remaining tokens,
        # excluding the last, which is guaranteed to be a stop token.
        sentence.extend(last_state[:-1])
        return sentence

    def next_state(self, state):
        return self.chain[state].sample()

    def generate_sentence(self):
        """Make a brand new sentence by taking a random walk down the chain.

        Returns:
            (str)
        """
        sentence = (token[0] if self.use_pos else token for token in self._generate_sentence())
        return TreebankWordDetokenizer().detokenize(sentence)


    def generate(self, n_sentences):
        return " ".join(self.generate_sentence() for _ in range(n_sentences))

    @staticmethod
    def print_nested_dict(d):
        """Prints a dictionary of unknown dimension depth-first.

        Args:
            d (dict): Any dictionary.
        """
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
