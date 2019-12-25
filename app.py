import marshal
from multiprocessing import Condition, Process, Queue, Pipe
from threading import Timer
from types import FunctionType
import pickle

from flask import Flask, url_for

from grams.grams import Histogram
from grams.markov import MC


import time

def make_app():

    def make_model():

        def _make_model(corpus, n_sentences=10):
            # global cv
            def _generate():
                markovchain = MC(corpus)
                return markovchain.generate
            generate = _generate()

            child_conn.send(generate(n_sentences))

            while True:
                if parent_conn.poll():
                    ## previously sent message got consumed
                    # send another
                    child_conn.send(generate(n_sentences))


        parent_conn, child_conn = Pipe(duplex=True)

        with open("res/the_adventures_of_sherlock_holmes.txt", "r") as f:
            f_out = f.read()
        make_process = Process(target=_make_model, args=(f_out,))
        make_process.start()

        return parent_conn, make_process

    app = Flask(__name__)
    parent_conn, make_process = make_model()

    @app.route("/")
    def home():
        if parent_conn.poll():
            return parent_conn.recv()
        return "loading..."

    return app


if __name__ == "__main__":
    app = make_app()
    app.run(debug=True, port=8080)
