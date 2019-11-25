# grams

A Python library for managing the histograms.

[![Build Status](https://travis-ci.com/escofresco/makeschool_fsp2_realtweets.svg?branch=master)](https://travis-ci.com/escofresco/makeschool_fsp2_realtweets)

[![codecov](https://codecov.io/gh/escofresco/makeschool_fsp2_realtweets/branch/master/graph/badge.svg)](https://codecov.io/gh/escofresco/makeschool_fsp2_realtweets)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install grams.

```bash
pip install grams
```

## Usage

```python
import grams

# Generate a histogram from a list of sentences
hist = grams.Histogram(['A sentence here.',
                        'A sentence there.',
                        'A sentence anywhere.'])

# Generate a histogram from a text file
with open('corpus.txt', 'r') as file:
    file_hist = grams.Histogram(file)

# Find the distance between histograms ∈ [0, 1]
similarity = hist.similarity(file_hist)

# Sample a random word weighted by its number of occurrences
word = hist.sample()

# Display word frequencies in the terminal
hist.visualize()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Lorem      : ▇▇▇▇ 4.00
# Ipsum      : ▇▇▇▇ 4.00
# is         : ▇ 1.00
# simply     : ▇ 1.00
# dummy      : ▇▇ 2.00
# text       : ▇▇ 2.00
# of         : ▇▇▇▇ 4.00
# the        : ▇▇▇▇▇▇ 6.00
# printing   : ▇ 1.00
# and        : ▇▇▇ 3.00
# typesetting: ▇▇ 2.00
# industry   : ▇ 1.00
# has        : ▇▇ 2.00
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Create a distribution object from code coverage data
cov = grams.Covergram("Documents/.coverage")

# Visualize code coverage data
cov.visualize()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# __init__.py       : ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 100.00
# grams.py          : ▇▇▇▇▇▇▇▇▇▇▇▇ 25.00
# hashtable.py      : ▇▇▇▇▇▇▇▇▇▇▇ 22.00
# linkedlist.py     : ▇▇▇▇▇▇▇ 15.00
# online.py         : ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 40.00
# stats.py          : ▇▇▇▇▇▇▇▇▇▇ 21.00
# termgraph.py      : ▇▇▇▇▇▇ 12.00
# utils.py          : ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 31.00
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# View statistics about a distribution
avg_code_cov = int(cov.mean)
frequency_standard_deviation = int(hist.std)
frequency_variance = int(hist.var)

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](LICENSE)
