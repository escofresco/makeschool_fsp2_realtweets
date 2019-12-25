from flask import Flask

from grams.grams import Histogram
from grams.markov import MC

app = Flask(__name__)

with open("res/the_adventures_of_sherlock_holmes.txt") as f:
    #histogram = Histogram(f)
    markovmodel = MC(str(f.read()))


@app.route("/")
def home():
    #return histogram.sample()
    return markovmodel.generate(10)


if __name__ == "__main__":
    app.run(debug=True)
