import setuptools
from distutils.command.sdist import sdist

with open("README.md", "r") as fh:
    long_description = fh.read()

class sdist_hg(sdist):

    user_options = sdist.user_options + [
            ('dev', None, "Add a dev marker")
            ]

    def initialize_options(self):
        sdist.initialize_options(self)
        self.dev = 0

    def run(self):
        if self.dev:
            print('asdfasdfsad')

    def initialize_options(self):
        sdist.initialize_options(self)
        self.dev = 0


setuptools.setup(
    name="grams",
    version="0.0.3",
    author="Jonasz Rice",
    author_email="jonaszakr@gmail.com",
    description="A package for managing histograms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/escofresco/makeschool_fsp2_realtweets",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "numpy==1.17.3",
        "dit==1.2.3",
        "colorama==0.4.1",
        "coverage==4.5.4",
    ],
    cmdclass={'sdist': sdist_hg})
