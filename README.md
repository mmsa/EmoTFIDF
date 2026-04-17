EmoTFIDF is an emotion detection library (Lexicon approach) based in the National Research Council Canada (NRC) and this package is for research purposes only. Source: [lexicons for research] (http://sentiment.nrc.ca/lexicons-for-research/)


#This library provides two types of emotions:

1- Lexicon based emotions which counting the frequency of the emotion based on the lexicon
2- Integrating TFIDF to add a context to the emotions.

#Installation


```python
pip install EmoTFIDF
```

## EmoTFIDF V2 (interpretable evidence layer)

V2 is a **parallel API** for research and tooling: it is meant as an interpretable lexical + TF-IDF **evidence** and **feature** module (explanations, richer vectors, prompt exports, and a light verifier for a proposed emotion label). It does **not** try to replace transformer baselines; see `docs/emotfidf_v2_notes.md` for design notes and suggested benchmarks.

**Negation** uses a fixed cue list and a short token window before each lexicon hit; contributions are scaled (and may flip sign) in a transparent, rule-based way. **Intensifiers / downtoners** apply simple multipliers in a short window before the affect token. **The verifier** surfaces lexical alignment only—it should not be read as semantic ground truth.

### V2 quick example

```python
from EmoTFIDF.v2 import EmoTFIDFv2

corpus = [
    "I am happy today and everything feels great.",
    "I am not happy today and everything feels wrong.",
    "I feel sad and disappointed about the news.",
]
v2 = EmoTFIDFv2()
v2.fit(corpus)

text = "I am very happy today!"
analysis = v2.analyze(text)
print(analysis.dominant_emotions)
print(analysis.to_dict()["negation_hits"])

print(v2.verify_label(text, "joy"))
print(v2.to_prompt_features(text))
```

You can also import the class from the package root: `from EmoTFIDF import EmoTFIDFv2`.

#List of emotions

-fear
-anger
-anticipation
-trust
-surprise
-positive
-negative
-sadness
-disgust
-joy


#Example of usage
##Get emotions from a sentence

```python
from EmoTFIDF.EmoTFIDF import EmoTFIDF

comment = "I had a GREAT week, thanks to YOU! I am very happy today."

emTFIDF = EmoTFIDF()

emTFIDF.set_text(comment)
print(emTFIDF.em_frequencies)
```


##Get emotions factorising TFIDF, you will need to add a context

Below is an example in pandas assuming you have a list of tweets/text and you would want to get emotions


```python
emTFIDF  = EmoTFIDF()
def getEmotionsTFIDF(s,emTFIDF):
  emTFIDF.set_text(s)
  emTFIDF.get_emotfidf()
  return emTFIDF.em_tfidf


emTFIDF.computeTFIDF(df['text'])
df['emotions'] = new_df.apply(lambda x: getEmotionsTFIDF(x['text'], emTFIDF), axis=1)#em_tfidf
df2 = df['emotions'].apply(pd.Series)
final_df = pd.concat([df,df2],axis=1)
```

#Plotting Emotion Distribution
You can visualize the distribution of emotions using the plot_emotion_distribution method:

```python
from EmoTFIDF.EmoTFIDF import EmoTFIDF

comment = "I had a GREAT week, thanks to YOU! I am very happy today."

emTFIDF = EmoTFIDF()
emTFIDF.set_text(comment)
emTFIDF.plot_emotion_distribution()
```

#Plotting Top TFIDF Words
To visualize the top N words by their TFIDF scores:
```python
import pandas as pd
from EmoTFIDF.EmoTFIDF import EmoTFIDF

# Assuming df is your DataFrame and it has a column 'text'
emTFIDF = EmoTFIDF()
emTFIDF.compute_tfidf(df['text'])
emTFIDF.plot_top_tfidf(top_n=20)


```
#Plotting TFIDF Weighted Emotion Scores
To visualize the TFIDF weighted emotion scores:
```python
from EmoTFIDF.EmoTFIDF import EmoTFIDF

comment = "I had a GREAT week, thanks to YOU! I am very happy today."

emTFIDF = EmoTFIDF()
emTFIDF.set_text(comment)
emTFIDF.get_emotfidf()
emTFIDF.plot_emotfidf()

```

##Update 1.4.2

Integrated Hybrid Method for Emotion Detection
New Features:

get_hybrid_emotions(text): Combines transformer-based and TFIDF weighted methods to get more accurate emotion scores.

```python

import pandas as pd
from EmoTFIDF.EmoTFIDF import EmoTFIDF

# Sample comments
comments = [
    "I had a GREAT week, thanks to YOU! I am very happy today.",
    "This is terrible. I'm so angry and sad right now.",
    "Looking forward to the weekend! Feeling excited and joyful.",
    "I am disgusted by the recent events. It's just awful.",
    "What a surprising turn of events! I didn't see that coming.",
]

# Create an instance of EmoTFIDF
emTFIDF = EmoTFIDF()

# Lists to store results
lexicon_emotions = []
tfidf_emotions = []
transformer_emotions = []
hybrid_emotions = []

# Process each comment and collect emotion frequencies and hybrid emotion scores
for comment in comments:
    emTFIDF.set_text(comment)
    lexicon_emotions.append(emTFIDF.em_frequencies)
    emTFIDF.compute_tfidf([comment])
    tfidf_emotions.append(emTFIDF.get_emotfidf())
    transformer_emotions.append(emTFIDF.get_transformer_emotions(comment))
    hybrid_emotions.append(emTFIDF.get_hybrid_emotions(comment))

# Create a DataFrame for the comments
df = pd.DataFrame(comments, columns=['text'])

# Add lexicon-based emotion frequencies to the DataFrame
df['lexicon_emotions'] = lexicon_emotions

# Add TFIDF weighted emotion scores to the DataFrame
df['tfidf_emotions'] = tfidf_emotions

# Add transformer-based emotion scores to the DataFrame
df['transformer_emotions'] = transformer_emotions

# Add hybrid emotion scores to the DataFrame
df['hybrid_emotions'] = hybrid_emotions

# Print the DataFrame with the new columns
print(df)

```
##Update 1.4.0

Integrated transformer-based models for advanced emotion detection.

New Features:
get_transformer_emotions(text): Uses a transformer model to get emotion scores.

plot_emotion_distribution(): Visualizes the distribution of emotions in the text using the transformer model.

```python
import pandas as pd
from EmoTFIDF.EmoTFIDF import EmoTFIDF

# Sample comments
comments = [
    "I had a GREAT week, thanks to YOU! I am very happy today.",
    "This is terrible. I'm so angry and sad right now.",
    "Looking forward to the weekend! Feeling excited and joyful.",
    "I am disgusted by the recent events. It's just awful.",
    "What a surprising turn of events! I didn't see that coming.",
]

# Create an instance of EmoTFIDF
emTFIDF = EmoTFIDF()

# Lists to store results
lexicon_emotions = []
transformer_emotions = []

# Process each comment and collect emotion frequencies and transformer emotion scores
for comment in comments:
    emTFIDF.set_text(comment)
    lexicon_emotions.append(emTFIDF.em_frequencies)
    transformer_emotions.append(emTFIDF.get_transformer_emotions(comment))

# Create a DataFrame for the comments
df = pd.DataFrame(comments, columns=['text'])

# Add lexicon-based emotion frequencies to the DataFrame
df['lexicon_emotions'] = lexicon_emotions

# Add transformer-based emotion scores to the DataFrame
df['transformer_emotions'] = transformer_emotions

# Print the DataFrame with the new columns
print(df)

# Visualize the transformer-based emotion scores for a sample comment
sample_comment = "I had a GREAT week, thanks to YOU! I am very happy today."
transformer_emotions = emTFIDF.get_transformer_emotions(sample_comment)

# Plot the transformer-based emotion scores
import matplotlib.pyplot as plt
import seaborn as sns

def plot_transformer_emotion_distribution(emotions):
    labels = list(emotions.keys())
    scores = list(emotions.values())

    plt.figure(figsize=(10, 5))
    sns.barplot(x=labels, y=scores)
    plt.title('Transformer-based Emotion Scores')
    plt.xlabel('Emotions')
    plt.ylabel('Scores')
    plt.show()

plot_transformer_emotion_distribution(transformer_emotions)

```

##Update 1.3.0

Introduced new plotting features to visualize the distribution of emotions, top TFIDF words, and TFIDF weighted emotion scores.

New Methods:
plot_emotion_distribution(): Visualizes the distribution of emotions in the text.

plot_top_tfidf(top_n=20): Visualizes the top N words by their TFIDF scores.

plot_emotfidf(): Visualizes the TFIDF weighted emotion scores.

These features enhance the interpretability of the emotion analysis by providing insightful visualizations.

##Update 1.0.7

Thanks to [artofchores](https://www.reddit.com/user/artofchores/), from Reddit for his feedback.


Added a set_lexicon_path option if you would like to use your own lexicon
Remember to keep the same structure as the original emotions lexicon which located [here](https://raw.githubusercontent.com/mmsa/EmoTFIDF/main/emotions_lex.json)
```
emTFIDF.set_lexicon_path("other_lexicon.json")
```

##Update 1.1.1

Updated the lexical db with some help from ChatGPT