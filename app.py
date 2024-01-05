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
# stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords = ['had', 'himself', 'needn', "needn't", 've', 'been', 'this', "you're", 'with', 'down', "isn't", 'me', 'who', 'as', 'isn', "weren't", 'i', 'why', 'was', 'are', 'off', 'against', 'during', 'own', "haven't", "that'll", "didn't", "hasn't", 'what', 'to', 'other', 'ours', 'myself', 'about', 'doing', 'same', 'haven', 'which', 'be', 'very', 'now', 'do', "shouldn't", 'll', 'weren', 'than', 'aren', "aren't", 'couldn', 'too', 'we', 'will', 'her', 'his', 'don', "hadn't", 'themselves', 'my', 'just', "mustn't", 'were', 'out', "don't", "won't", 'until', 'yourself', 'both', 'doesn', 'above', 'your', 'you', 'for', 'before', 'between', 'is', 's', 'itself', 'the', 'of', 'having', 'them', 'up', 'here', 'each', 'because', 'hasn', 'by', 'mightn', 'not', 'and', 'he', 'him', 'does', 'in', 'at', 'she', "you'll", 'when', 'some', 'hadn', "wouldn't", 'over', 'few', 'd', 'wouldn', 'there', "should've", 'while', "it's", 're', 'under', 'ain', 'into', 'won', 'm', 'or', 'then', 'yourselves', 'most', 'nor', 'so', 'if', 'further', 'after', 'theirs', "shan't", 'but', 'again', 'herself', 'didn', 'ma', 'ourselves', 'once', 'through', 'o', "doesn't", 'wasn', "mightn't", 'such', 'have', 'below', 'an', 'shouldn', 'these', 'a', 'their', "wasn't", 'did', "she's", 'whom', 'any', 'only', 'should', 'that', 'our', 'am', 't', 'on', 'can', 'yours', 'where', 'those', 'more', 'being', 'all', "you've", 'its', 'mustn', 'hers', 'it', 'they', 'has', 'no', 'shan', "you'd", "couldn't", 'y', 'from', 'how']

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