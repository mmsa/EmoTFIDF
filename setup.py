#!/usr/bin/env python
"""
Author: mmsa12
"""

import setuptools
from setuptools.command.install import install as _install

with open("README.md", "r") as fh:
    long_description = fh.read()


class Install(_install):
    def run(self):
        _install.do_egg_install(self)
        import nltk
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('stopwords')


setuptools.setup(
    name="EmoTFIDF",
    version="1.2.0",
    author="mmsa12",
    author_email="mmsa12@gmail.com",
    description="A library to extract emotions using two methods: 1) Using lexicon-based counting frequency of emotion, "
                "2) Integrating TFIDF to add context.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mmsa/emotfidf",
    packages=setuptools.find_packages(),
    install_requires=['nltk'],
    include_package_data=True,
    setup_requires=['nltk'],
    tests_require=['pytest'],
    extras_require={
        'dev': ['pytest'],
    },
    package_data={'emotfidf': ['emotions_lex.json']},
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',
)
