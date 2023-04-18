EmoTFIDF is an emotion detection library (Lexicon approach) based in the National Research Council Canada (NRC) and this package is for research purposes only. Source: [lexicons for research] (http://sentiment.nrc.ca/lexicons-for-research/)


#This library provides two types of emotions:

1- Lexicon based emotions which counting the frequency of the emotion based on the lexicon
2- Integrating TFIDF to add a context to the emotions.

#Installation


```python
pip install EmoTFIDF
```

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
from emotfidf import EmoTFIDF

comment = "I had a GREAT week, thanks to YOU! I am very happy today."

emTFIDF  = EmoTFIDF()

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

##Update 1.0.7

Thanks to [artofchores](https://www.reddit.com/user/artofchores/), from Reddit for his feedback.


Added a set_lexicon_path option if you would like to use your own lexicon
Remember to keep the same structure as the original emotions lexicon which located [here](https://raw.githubusercontent.com/mmsa/EmoTFIDF/main/emotions_lex.json)
```
emTFIDF.set_lexicon_path("other_lexicon.json")
```

##Update 1.1.1

Updated the lexical db with some help from ChatGPT