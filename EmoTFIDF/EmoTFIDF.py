import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import string
import nltk
from nltk.corpus import stopwords
import urllib.request
import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the tokenizer and model for transformer-based emotion detection
tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

def process_message(message):
    """
    Process a text message by tokenizing, removing stop words, numbers, small words, and punctuation.

    Args:
        message (str): The input text message to process.

    Returns:
        str: The processed text message.
    """
    words = nltk.word_tokenize(message.lower())
    words = [w for w in words if len(w) > 3]  # remove small words
    sw = stopwords.words('english')
    words = [word for word in words if not word.isnumeric()]
    words = [word for word in words if word not in sw]
    words = [word for word in words if word not in string.punctuation]
    return ' '.join(words)

def get_emotions(self):
    """
    Extract emotions from the processed text based on a predefined lexicon.

    This function updates the following attributes of the EmoTFIDF object:
    - em_list: A list of emotions found in the text.
    - em_dict: A dictionary mapping words to their corresponding emotions.
    - emotion_scores: A frequency count of each emotion found.
    - em_frequencies: The percentage of each emotion relative to the total number of emotions found.
    """
    em_list = []
    em_dict = dict()
    em_frequencies = Counter()
    lexicon_keys = self.lexicon.keys()
    for word in self.words:
        if word in lexicon_keys:
            if isinstance(self.lexicon[word], list):
                emotions_found = list(set(self.lexicon[word]))
                emotions_found = list(filter(None, emotions_found))
                if emotions_found is not None:
                    if 'negative' in emotions_found:
                        emotions_found.remove('negative')
                    elif 'positive' in emotions_found:
                        emotions_found.remove('positive')
                    em_list.extend(emotions_found)
                    em_dict.update({word: self.lexicon[word]})
    for word in em_list:
        em_frequencies[word] += 1
    sum_values = sum(em_frequencies.values())
    em_percent = {key: 0.0 for key in labels}
    for key in em_frequencies.keys():
        em_percent.update({key: float(em_frequencies[key]) / float(sum_values)})
    self.em_list = em_list
    self.em_dict = em_dict
    self.emotion_scores = dict(em_frequencies)
    self.em_frequencies = em_percent

class EmoTFIDF:
    """
    A class to process text, extract emotions using a lexicon, and compute TF-IDF scores for words.

    Attributes:
        lexicon (dict): A dictionary containing words as keys and corresponding emotions as values.
    """
    with urllib.request.urlopen("https://raw.githubusercontent.com/mmsa/EmoTFIDF/main/EmoTFIDF/emotions_lex.json") as url:
        lexicon = json.loads(url.read().decode())

    def set_text(self, text):
        """
        Set and process the input text for emotion extraction and TF-IDF computation.

        Args:
            text (str): The input text to process.
        """
        self.text = process_message(text)
        self.words = list(nltk.word_tokenize(self.text))
        self.sentences = list(nltk.sent_tokenize(self.text))
        get_emotions(self)

    def set_lexicon_path(self, path):
        """
        Set the path to a custom lexicon file.

        Args:
            path (str): The file path to the custom lexicon in JSON format.
        """
        self.path = path
        if path and path.strip():
            with open(path) as jsonfile:
                self.lexicon = json.load(jsonfile)
        else:
            with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/mmsa/EmoTFIDF/main/emotions_lex.json") as url:
                self.lexicon = json.loads(url.read().decode())

    def compute_tfidf(self, docs):
        """
        Compute TF-IDF scores for a collection of documents.

        Args:
            docs (list of str): A list of documents to process.

        Updates:
            self.tfid (pd.DataFrame): A DataFrame containing TF-IDF scores for the documents.
            self.vectorizer (TfidfVectorizer): The TF-IDF vectorizer used for computation.
            self.feature_names (list of str): The feature names (terms) used in the TF-IDF vectorizer.
        """
        stop_words = stopwords.words('english')
        vectorizer = TfidfVectorizer(max_features=200, stop_words=stop_words, token_pattern=r'(?u)\b[A-Za-z]+\b')
        vectors = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()
        dense = vectors.todense()
        denselist = dense.tolist()
        df = pd.DataFrame(denselist, columns=feature_names)
        self.tfid = df
        self.vectorizer = vectorizer
        self.feature_names = feature_names

    def get_ifidf_for_words(self):
        """
        Compute TF-IDF scores for words in the processed text.

        Updates:
            self.ifidf_for_words (dict): A dictionary mapping words to their TF-IDF scores.
        """
        tfidf_matrix = self.vectorizer.transform([self.text]).todense()
        feature_index = tfidf_matrix[0, :].nonzero()[1]
        tfidf_scores = zip([self.feature_names[i] for i in feature_index], [tfidf_matrix[0, x] for x in feature_index])
        self.ifidf_for_words = dict(tfidf_scores)

    def get_emotfidf(self):
        """
        Compute emotion scores factorized by TF-IDF weights.

        Updates:
            self.em_tfidf (dict): A dictionary containing the TF-IDF weighted emotion scores.
        """
        self.get_ifidf_for_words()
        em_percent = {key: 0.0 for key in labels}
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
            em_percent.update({key: round(float(em_frequencies[key]) / float(sum_values), 3)})
        self.em_tfidf = em_percent

    def get_transformer_emotions(self, text):
        """
        Get emotion scores using a transformer model.

        Args:
            text (str): The input text to process.

        Returns:
            dict: A dictionary mapping emotions to their scores.
        """
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        scores = outputs[0][0].detach().numpy()
        scores = (scores - scores.min()) / (scores.max() - scores.min())  # Normalize to 0-1
        return {labels[i]: scores[i] for i in range(len(scores))}

    def get_hybrid_emotions(self, text):
        """
        Get emotion scores using a hybrid method combining transformer and TF-IDF scores.

        Args:
            text (str): The input text to process.

        Returns:
            dict: A dictionary mapping emotions to their scores.
        """
        tfidf_emotions = self.get_emotfidf()
        transformer_emotions = self.get_transformer_emotions(text)
        hybrid_emotions = {label: (tfidf_emotions.get(label, 0) + transformer_emotions.get(label, 0)) / 2 for label in labels}
        return hybrid_emotions

    def plot_emotion_distribution(self):
        """
        Plot the distribution of emotions found in the text.
        """
        if not hasattr(self, 'em_frequencies'):
            raise AttributeError("Emotion frequencies not found. Please run set_text() method first.")

        emotions = list(self.em_frequencies.keys())
        frequencies = list(self.em_frequencies.values())

        plt.figure(figsize=(10, 5))
        sns.barplot(x=emotions, y=frequencies)
        plt.title('Emotion Distribution')
        plt.xlabel('Emotions')
        plt.ylabel('Frequency')
        plt.show()

    def plot_top_tfidf(self, top_n=20):
        """
        Plot the top N words based on their TF-IDF scores.

        Args:
            top_n (int): Number of top words to display. Default is 20.
        """
        if not hasattr(self, 'tfid'):
            raise AttributeError("TF-IDF data not found. Please run compute_tfidf() method first.")

        top_words = self.tfid.sum().sort_values(ascending=False).head(top_n)

        plt.figure(figsize=(10, 5))
        sns.barplot(x=top_words.index, y=top_words.values)
        plt.title(f'Top {top_n} Words by TF-IDF Score')
        plt.xlabel('Words')
        plt.ylabel('TF-IDF Score')
        plt.xticks(rotation=45)
        plt.show()

    def plot_emotfidf(self):
        """
        Plot the TF-IDF weighted emotion scores.
        """
        if not hasattr(self, 'em_tfidf'):
            raise AttributeError("TF-IDF weighted emotion scores not found. Please run get_emotfidf() method first.")

        emotions = list(self.em_tfidf.keys())
        scores = list(self.em_tfidf.values())

        plt.figure(figsize=(10, 5))
        sns.barplot(x=emotions, y=scores)
        plt.title('TF-IDF Weighted Emotion Scores')
        plt.xlabel('Emotions')
        plt.ylabel('Weighted Score')
        plt.show()
