# grams.stats
#!/usr/bin/env python3
"""Module for implementations of Markov models.

Attributes:
    Markov: A generic Markov model class.
    MC: The most basic Markov model,
"""

from collections import Counter, defaultdict, deque, namedtuple
from copy import deepcopy
from functools import reduce
from itertools import chain
import os
import pickle

from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from .grams import Gram, Histogram
from .stats import FreqDist
from .utils import capture_stdout

__all__ = ["Markov", "MC", "HMM"]


class Markov:
    """Base class for Markov Model implementations which enforces three
    assumptions:
    `The Markov assumption`_: the next state only depends on the
        current state and not on the past). This means the conditional
        probability distribution for the system at the next step doesn't depend on
        the state of the system at previous steps.
    `The stationarity assumption`_: When a transition between two states occurs,
        time is independent.
    `The observation independence assumption`_: The current observation is
        independent of other observations.

    .. _The Markov Assumption
        https://ccrma.stanford.edu/~kglee/pubs/klee-ismir06.pdf
    .. _The stationarity assumption
    .. _The observation independence assumption
        https://learning.oreilly.com/library/view/markov-processes-for/9780124077959/xhtml/CHP014.html
    """
    __slots__ = ()


class MC(Markov):
    """This is the `Metropolis-Hastings`_ implementation of the Markov
    Chain Monte Carlo (MCMC). An ensemble of chains is built from `visible transitions
    between states that we can't control`_. A random walk is taken across states
    by sampling randomly from the conditional probability distribution. There
    are two big drawbacks to MCMCs. The first is autocorrelation. Although
    samples between states are independent, they're still correlated. Even if,
    on the aggregate, our model produces something statistically similar
    to the data it was trained on, nearby samples can clump together, creating
    repetitive sequences. The second drawback has to do with unevenly
    distributed data. If our chain was built from written language, which is
    unevenly distributed according to Zipf's law, the initial samples would likely
    be from low-density distributions, creating results that don't represent the
    data being trained on. Burn-in solves this by throwing away the first N
    samples. One idea would be to borrow the idea of temperature from simulated
    annealing. Starting out very hot, the model would be readily discard tokens.
    As the model settled into a "low energy state", it would apply more value to
    its observations.

    Attributes:
        order (int): Number of prior linguistic units to keep in memory. Making
            this number too high results in overfitting, producing results
            that are verbatim taking from the training data. But, making it too
            low loses information about context, such as rare events with high
            certainty; "modus" is a rare word, but it is almost certainly
            followed by "operandi".
        chain (dict): Mapping of states to transitions (histograms).

    .. _Metropolis-Hastings
        https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
    .. _visible transitions between states that we can't control
        https://learning.oreilly.com/library/view/markov-processes-for/9780124077959/xhtml/CHP014.html
    """
    __slots__ = ("order", "memory", "use_pos", "start_token", "stop_token",
                 "start_state", "chain")

    START_TOKEN = "<|~~START~~|>"
    STOP_TOKEN = "<|~~STOP~~|>"

    def __init__(self, corpus, order=1, use_pos=True):
        self.order = order
        self.memory = self.order + 2
        self.use_pos = use_pos

        # use start and stop to signify the beginning and end of a sentence.
        self.start_token = MC.START_TOKEN
        self.stop_token = MC.STOP_TOKEN
        if use_pos:
            self.start_token = (self.start_token, self.start_token)
            self.stop_token = (self.stop_token, self.stop_token)

        self.start_state = (self.start_token,) * (self.memory - 1)

        # use the corpus and a word order
        self.chain = dict(self._make_chain(self._make_transitions(corpus)))

    def __str__(self):
        res = []
        for state, transition_hist in self.chain.items():
            res.extend((f"\n{'~'*10} {state} {'~'*10}",
                        capture_stdout(transition_hist.show)))
        return "\n".join(res)

    def __eq__(self, other):
        for attr in MC.__slots__:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def _make_transitions(self, corpus):
        transitions = defaultdict(Counter)

        # Fill cache with start tokens. Since a start token is added at the
        # beginning of each iteration, don't add the last two.
        cache = deque(self.start_state[:-1], maxlen=self.memory)

        for sentence in (Gram.pos_sents(corpus)
                         if self.use_pos else Gram.sents(corpus)):
            ## Depending on if pos_tags are being used, look through either
            ## (token, pos) pairs or just the tokens
            for word_pos in chain((self.start_token,), sentence,
                                  (self.stop_token,)):
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
        sentence = (token[0] if self.use_pos else token
                    for token in self._generate_sentence())
        return TreebankWordDetokenizer().detokenize(sentence)

    def generate(self, n_sentences):
        return " ".join(self.generate_sentence() for _ in range(n_sentences))

    def save(self, filename="markovchain.pkl"):
        """Saves an object to a pickle.

        Args:
            obj: Any object intended to be saved.
            filename: Name of the pickle to save to, ending in .pkl.
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    def print_chain(self):
        self.print_nested_dict(self.chain)

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


class HMM(Markov):
    """This is a doubly stochastic finite state machine that uses a sequence of
    observations (a policy) to
    evaluate underlying processes. Transitions are neither visible to nor
    controllable by an agent. State processes (as well as transition
    probabilities) are therefore "hidden" and are evaluated through observation
    processes (based on emission probabilities). The opaqueness of HMMs cause
    three problems to arise.

    `The evaluation problem`_: How do we find out if the observations generated
        happen with the correct probability? Possible solution: forward
        algorithm and backward algorithm.
    `The decoding problem`_: How to we find the most likely sequence of states
        at any point in time? Possible solution: Viterbi algorithm.
    `The learning problem`_: Given a sequence of observations, how to we find
        the parameters that would model that outcome. Possible solution:
        Baum-Welch algorithm.

    .. _The evaluation problem
    .. _The decoding problem
    .. _The learning problem
        https://learning.oreilly.com/library/view/markov-processes-for/9780124077959/xhtml/CHP014.html
    """

    __slots__ = ()


if __name__ == "__main__":
    txt = """Arfool is the best atrese I've ever tasted.
            Yea, I need to blash arfool.
            """
    mc = MC(txt)
    mc.save("small_markov_chain.pkl")
