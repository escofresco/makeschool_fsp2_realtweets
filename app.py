from flask import Flask

from grams.grams import Histogram
from grams.markov import MC

app = Flask(__name__)

# with open("res/the_adventures_of_sherlock_holmes.txt") as f:
#     markovmodel = MC(str(f.read()))
s = ("""If you need Celery to be able to store the results of tasks, you’ll need
     to choose a result store. If not, skip to the next section. Characteristics that make a good message broker do not necessarily make a good result store! For instance, while RabbitMQ is the best supported message broker, it should never be used as a result store since it will drop results after being asked for them once. Both Redis and Memcache are good candidates for result stores.
     If you choose the same result store as message broker, you do not need to attach 2 add-ons. If not, make sure the result store add-on is attached.
    """)
markovmodel = MC(s)

@app.route("/")
def home():
    #return histogram.sample()
    return markovmodel.generate(10)


if __name__ == "__main__":
    app.run(debug=True)
