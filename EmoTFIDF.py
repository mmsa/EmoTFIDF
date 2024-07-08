#!/usr/bin/env python
"""
EmoTFIDF: A library for detecting emotions in text using TF-IDF and lexicons.
@author: mmsa12
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import string
import nltk
from nltk.corpus import stopwords
import urllib.request
import json

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')


class EmoTFIDF:
    """Lexicon source is (C) 2016 National Research Council Canada (NRC) and library is for research purposes only.
    Source: http://sentiment.nrc.ca/lexicons-for-research/
    """

    def __init__(self, lexicon_url=None):
        self.lexicon_url = lexicon_url or "https://raw.githubusercontent.com/mmsa/EmoTFIDF/main/emotions_lex.json"
        self.load_lexicon()
        self.tfid = None
        self.vectorizer = None
        self.feature_names = None
        self.ifidf_for_words = None
        self.em_tfidf = None

    def load_lexicon(self):
        try:
            with urllib.request.urlopen(self.lexicon_url) as url:
                self.lexicon = json.loads(url.read().decode())
        except Exception as e:
            print(f"Error loading lexicon: {e}")
            self.lexicon = {}

    def set_lexicon_path(self, path):
        try:
            if path and path.strip():
                with open(path) as jsonfile:
                    self.lexicon = json.load(jsonfile)
            else:
                self.load_lexicon()
        except Exception as e:
            print(f"Error setting lexicon path: {e}")
            self.lexicon = {}

    def set_text(self, text):
        self.text = self.process_message(text)
        self.words = list(nltk.word_tokenize(self.text))
        self.sentences = list(nltk.sent_tokenize(self.text))
        self.get_emotions()

    @staticmethod
    def process_message(message):
        words = nltk.word_tokenize(message.lower())
        words = [w for w in words if len(w) > 3]  # remove small words
        sw = stopwords.words('english')
        words = [word for word in words if not word.isnumeric() and word not in sw and word not in string.punctuation]
        return ' '.join(words)

    def get_emotions(self):
        em_list = []
        em_dict = {}
        em_frequencies = Counter()
        for word in self.words:
            if word in self.lexicon:
                emotions_found = list(set(filter(None, self.lexicon[word])))
                if 'negative' in emotions_found:
                    emotions_found.remove('negative')
                if 'positive' in emotions_found:
                    emotions_found.remove('positive')
                em_list.extend(emotions_found)
                em_dict[word] = self.lexicon[word]
        for word in em_list:
            em_frequencies[word] += 1
        sum_values = sum(em_frequencies.values())
        em_percent = {emotion: 0.0 for emotion in ['fear', 'anger', 'anticipation', 'trust', 'surprise', 'sadness', 'disgust', 'joy']}
        for key in em_frequencies:
            em_percent[key] = float(em_frequencies[key]) / float(sum_values)
        self.em_list = em_list
        self.em_dict = em_dict
        self.emotion_scores = dict(em_frequencies)
        self.em_frequencies = em_percent

    def compute_tfidf(self, docs):
        stop_words = stopwords.words('english')
        vectorizer = TfidfVectorizer(max_features=200, stop_words=stop_words, token_pattern=r'(?u)\b[A-Za-z]+\b')
        vectors = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()
        dense = vectors.todense()
        denselist = dense.tolist()
        self.tfid = pd.DataFrame(denselist, columns=feature_names)
        self.vectorizer = vectorizer
        self.feature_names = feature_names

    def get_ifidf_for_words(self):
        tfidf_matrix = self.vectorizer.transform([self.text]).todense()
        feature_index = tfidf_matrix[0].nonzero()[1]
        tfidf_scores = zip([self.feature_names[i] for i in feature_index], [tfidf_matrix[0, x] for x in feature_index])
        self.ifidf_for_words = dict(tfidf_scores)

    def get_emotfidf(self):
        self.get_ifidf_for_words()
        em_percent = {emotion: 0.0 for emotion in ['fear', 'anger', 'anticipation', 'trust', 'surprise', 'sadness', 'disgust', 'joy']}
        em_frequencies = Counter(self.em_list)
        for e in self.em_dict:
            if e in self.ifidf_for_words:
                tfidf_weight = self.ifidf_for_words[e]
                for a in self.em_dict[e]:
                    new_fre = round(em_frequencies[a] / tfidf_weight, 2)
                    em_frequencies[a] = new_fre
        sum_values = sum(em_frequencies.values())
        for key in em_frequencies:
            em_percent[key] = round(float(em_frequencies[key]) / float(sum_values), 3)
        self.em_tfidf = em_percent
