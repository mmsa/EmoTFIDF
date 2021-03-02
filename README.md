EmoTFIDF is an emotion detection library (Lexicon approach) based in the National Research Council Canada (NRC) and this package is for research purposes only. Source: [lexicons for research] (http://sentiment.nrc.ca/lexicons-for-research/)


#This library provides two types of emotions:

1- Lexicon based emotions which counting the frequency of the emotion based on the lexicon
2- Integrating TFIDF to add a context to the emotions.

<h1>Installation</h1>

```python
pip install EmoTFIDF
```


<h2>List of emotions:</h2>

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


<h2>Example of usage:</h2?

<h3>Get emotions from a sentence</h3>
```python
from emotfidf import EmoTFIDF

comment = "I had a GREAT week, thanks to YOU! If you need anything, please reach out."

emTFIDF  = EmoTFIDF()

emTFIDF.set_text(comment)
emTFIDF.get_emotions()
```

returns lists of emotions

#Return words list.

```python
emTFIDF.words
```

<h3>Get emotions factorising TFIDF, you will need to add a context<3>

Below is an example in pandas assuming you have a list of tweets/text and you would want to get emotions
```python
emTFIDF  = EmoTFIDF()
def getEmotionsTFIDF(s,emTFIDF):
  emTFIDF.set_text(s)
  emTFIDF.get_emotfidf()
  return emTFIDF.em_frequencies


emTFIDF.computeTFIDF(df['text'])
df['emotions'] = new_df.apply(lambda x: getEmotionsTFIDF(x['text'], emTFIDF), axis=1)#em_tfidf
df2 = df['emotions'].apply(pd.Series)
final_df = pd.concat([df,df2],axis=1)
```
