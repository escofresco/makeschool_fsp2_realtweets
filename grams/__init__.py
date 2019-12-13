import os

import nltk

from .grams import *
from .online import *
from .stats import *
from .utils import *

# add the `nltk_data` folder to nltk's list of directories to find data in
nltk.data.path.append(os.path.join(os.getcwd(), "grams/nltk_data/"))
