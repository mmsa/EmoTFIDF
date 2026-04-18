#!/usr/bin/env python
"""Setuptools metadata for EmoTFIDF (PyPI)."""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EmoTFIDF",
    version="1.5.0",
    author="mmsa12",
    author_email="mmsa12@gmail.com",
    description=(
        "Lexicon + TF-IDF emotion features (V1), hybrid transformer support, and V2 "
        "interpretable lexical evidence (EmoTFIDFv2). Lexicon: research use only."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mmsa/emotfidf",
    project_urls={
        "Source": "https://github.com/mmsa/emotfidf",
        "Tracker": "https://github.com/mmsa/emotfidf/issues",
    },
    # Ship only the library; ``experiments/`` is local benchmark code, not a distribution package.
    packages=["EmoTFIDF", "EmoTFIDF.evidence"],
    install_requires=[
        "numpy>=1.19.0",
        "nltk>=3.6",
        "scikit-learn>=1.0.0",
        "pandas>=1.2.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "torch>=1.9.0",
        "transformers>=4.20.0",
    ],
    include_package_data=True,
    package_data={"EmoTFIDF": ["emotions_lex.json"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    keywords="emotion NLP TF-IDF lexicon transformers interpretability",
)
