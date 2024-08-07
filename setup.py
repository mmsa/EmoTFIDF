#!/usr/bin/env python
"""
@author: mmsa12
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
    version="1.4.2",
    author="mmsa12",
    author_email="mmsa12@gmail.com",
    description="A library to extract emotions using three methods: 1- Lexicon based, counting frequency of emotion, "
                "2- Integrating TFIDF to add context, and 3- Hybrid transformer-TFIDF approach. "
                "Note that lexicon license is for research purposes only.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mmsa/emotfidf",
    packages=setuptools.find_packages(),
    install_requires=[
        'nltk',
        'scikit-learn',
        'pandas',
        'matplotlib',
        'seaborn',
        'torch',
        'transformers'
    ],
    include_package_data=True,
    py_modules=["EmoTFIDF"],
    setup_requires=['nltk'],
    package_data={'EmoTFIDF': ['emotions_lex.json']},
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
