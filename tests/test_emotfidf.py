from EmoTFIDF import EmoTFIDF


def test_basic_emotion_extraction():
    text = "I am so happy today!"
    emTFIDF = EmoTFIDF()
    emTFIDF.set_text(text)
    emotions = emTFIDF.em_frequencies
    assert 'joy' in emotions and emotions['joy'] > 0
