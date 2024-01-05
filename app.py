import pickle
import nltk
from tensorflow.keras.models import load_model
from nltk.stem import PorterStemmer
import re
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import streamlit as st

model = load_model('model1.h5')
# nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

model = load_model('model1.h5')
lb = pickle.load(open('lb.pkl','rb'))
def sentence_cleaning(sentence):
    stemmer = PorterStemmer()
    corpus = []
    text = re.sub("[^a-zA-Z]", " ", sentence)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    text = " ".join(text)
    corpus.append(text)
    one_hot_word = [one_hot(input_text=word, n=11000) for word in corpus]
    pad = pad_sequences(sequences=one_hot_word, maxlen=300, padding='pre')
    return pad

sentence = sentence_cleaning("i am ever feeling nostalgic about the fireplace i will know that it is still on the property")
result = lb.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]
proba =  np.max(model.predict(sentence))

st.title("Human Emotions Detection App")
st.write("=================================================")
st.write("['Joy,'Fear','Neutral','Sad']")
st.write("=================================================")

# taking input from user
user_input = st.text_input("Enter your text here:")

if st.button("Predict"):
    sentence = sentence_cleaning(user_input)
    result = lb.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]
    st.write("Predicted Emotion:", result)
    st.write("Probability:", proba)