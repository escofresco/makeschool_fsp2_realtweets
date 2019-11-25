from flask import Flask

from grams.grams import Histogram

app = Flask(__name__)

with open("res/the_adventures_of_sherlock_holmes.txt") as f:
    histogram = Histogram(f)


@app.route("/")
def home():
    return histogram.rand_word()
