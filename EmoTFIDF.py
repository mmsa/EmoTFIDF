#!/usr/bin/env python
"""
@author: mmsa12
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import json
import string, nltk
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')


def process_message(message):
    words = nltk.word_tokenize(message.lower())
    words = [w for w in words if len(w) > 3]  # remove small words
    # stop words
    sw = stopwords.words('english')
    words = [word for word in words if word.isnumeric() == False]
    words = [word for word in words if word not in sw]
    # remove punctuations
    words = [word for word in words if word not in string.punctuation]
    return ' '.join(words)


def get_emotions(self):
    # get emotions
    em_list = []
    em_dict = dict()
    em_frequencies = Counter()
    lexicon_keys = self.lexicon.keys()
    for word in self.words:
        if word in lexicon_keys:
            em_list.extend(self.lexicon[word])
            em_dict.update({word: self.lexicon[word]})
    for word in em_list:
        em_frequencies[word] += 1
    sum_values = sum(em_frequencies.values())
    em_percent = {'fear': 0.0, 'anger': 0.0, 'anticipation': 0.0, 'trust': 0.0, 'surprise': 0.0, 'positive': 0.0,
                  'negative': 0.0, 'sadness': 0.0, 'disgust': 0.0, 'joy': 0.0}
    for key in em_frequencies.keys():
        em_percent.update({key: float(em_frequencies[key]) / float(sum_values)})
    self.em_list = em_list
    self.em_dict = em_dict
    self.emotion_scores = dict(em_frequencies)
    self.em_frequencies = em_percent


class EmoTFIDF:
    """Lexicon source is (C) 2016 National Research Council Canada (NRC) and library is for research purposes only.  Source: http://sentiment.nrc.ca/lexicons-for-research/"""

    with open('emotions_lex.json') as jsonfile:
        lexicon = json.load(jsonfile)

    def set_text(self, text):
        self.text = process_message(text)
        self.words = list(nltk.word_tokenize(self.text))
        self.sentences = list(nltk.sent_tokenize(self.text))
        get_emotions(self)
        top_emotions(self)

    def computeTFIDF(self, docs):
        vectorizer = TfidfVectorizer(max_features=200, stop_words=stopwords.words('english'),
                                     token_pattern=r'(?u)\b[A-Za-z]+\b')
        vectors = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names()
        dense = vectors.todense()
        denselist = dense.tolist()
        df = pd.DataFrame(denselist, columns=feature_names)
        self.tfid = df
        self.vectorizer = vectorizer
        self.feature_names = feature_names

    def get_ifidf_for_words(self):
        tfidf_matrix = self.vectorizer.transform([self.text]).todense()
        feature_index = tfidf_matrix[0, :].nonzero()[1]
        tfidf_scores = zip([self.feature_names[i] for i in feature_index], [tfidf_matrix[0, x] for x in feature_index])
        self.ifidf_for_words = dict(tfidf_scores)

    def get_emotfidf(self):
        self.get_ifidf_for_words()
        em_percent = {'fear': 0.0, 'anger': 0.0, 'anticipation': 0.0, 'trust': 0.0, 'surprise': 0.0, 'positive': 0.0,
                      'negative': 0.0, 'sadness': 0.0, 'disgust': 0.0, 'joy': 0.0}
        em_frequencies = Counter()
        for word in self.em_list:
            em_frequencies[word] += 1
        for e in self.em_dict.keys():
            if e in self.ifidf_for_words:
                tfidf_weight = self.ifidf_for_words[e]
                for a in self.em_dict[e]:
                    new_fre = round(em_frequencies[a] / tfidf_weight, 2)
                    em_frequencies[a] = new_fre
        sum_values = sum(em_frequencies.values())
        for key in em_frequencies.keys():
            em_percent.update({key: float(em_frequencies[key]) / float(sum_values)})
        self.em_tfidf = em_percent
