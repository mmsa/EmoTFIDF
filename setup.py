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
    version="1.0.0",
    author="mmsa12",
    author_email="mmsa12@gmail.com",
    description="A library to extract emotions using two methods, 1- Using lexicon based, counting frequency of emotion"
                "2- Integrating TFIDF to add a context"
                "Note that lexicon license is for research purposes only.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mmsa12/emotfidf",
    packages=setuptools.find_packages(),
    install_requires=['nltk'],
    include_package_data=True,
    py_modules=["emotfidf"],
    setup_requires=['nltk'],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',
)