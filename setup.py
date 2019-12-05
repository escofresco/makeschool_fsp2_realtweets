import os
from os.path import exists
from stat import S_IEXEC, S_IRUSR, S_IWUSR

from distutils.command.sdist import sdist
from setuptools import find_packages, setup

from grams import recur_chmod

with open("README.md", "r") as fh:
    long_description = fh.read()


class sdist_hg(sdist):
    """Subclass sdist as a convenient way to do some development steps.
    run `python setup.py sdist --prepvc`
    a slight copy of sdist docs example: https://tinyurl.com/rw853kn
    """

    user_options = sdist.user_options + [
        ('prepvc', None, "Do  some things to prepare for github "
         "version control")
    ]

    def initialize_options(self):
        sdist.initialize_options(self)
        self.prepvc = 0

    def run(self):
        """Override super().run()"""
        if self.prepvc:
            self.prep_git_versioning()

    def prep_git_versioning(self):
        """Do some things to make this project fully ready for
        development"""
        # pre-push to .git/hooks/, found by github, which then runs
        # .githooks/pre-push
        # this is how pre-push script is being kept in version control

        with open(".git/hooks/pre-push", "w") as hookscript:
            hookscript.write("#!/bin/sh\n" "sh .githooks/pre-push")

        # change permissions to execute by owner so github can use it, write
        # by owner in case the above is called on existing file,
        # and read in case we want to ever see it.
        # equivalent to chmod +x; apply to .git/hooks/pre-push
        ## adapted from https://stackoverflow.com/a/12792002/8011811
        os.chmod(".git/hooks/pre-push", S_IWUSR | S_IEXEC | S_IRUSR)

        # apply chomod +x recursively to .githooks/
        #recur_chmod("./.githooks")


setup(name="grams",
      version="0.0.3",
      author="Jonasz Rice",
      author_email="jonaszakr@gmail.com",
      description="A package for managing histograms.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/escofresco/makeschool_fsp2_realtweets",
      packages=find_packages(),
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
