import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('stopwords')
STOPWORDS = nltk.corpus.stopwords.words('english')

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenization
    filtered_tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalnum() and not w in STOPWORDS]  # Remove stopwords and lemmatize
    return ' '.join(filtered_tokens)

vocab_size = 5000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

# Layout Width
st. set_page_config(layout="centered")
st.image('header_image.jpg')
st.header('News Classificatoion using RNN', divider='rainbow')
st.subheader('Please enter the blog post in the below area -')
blog = st.text_area("Blog post:")
st.write(f'You wrote {len(blog)} characters.')    
if st.button("Predict", type="primary"):
    if blog != "":
        path_to_model = './models/bbc_text_classification_6.h5'
        model = load_model(path_to_model, compile=False)
        # Preprocess the single news row
        preprocessed_single_news = preprocess_text(blog)

        # Load tokenizer used during training
        tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        tokenizer.fit_on_texts([preprocessed_single_news])

        # Convert text to sequence
        sequence_single_news = tokenizer.texts_to_sequences([preprocessed_single_news])

        # Pad sequence
        padded_single_news = pad_sequences(sequence_single_news, maxlen=max_length)

        # Predict
        predicted_prob = model.predict(padded_single_news)
        labels = ['tech', 'sport', 'business', 'entertainment', 'politics']
        ind = np.array(predicted_prob).argmax(axis=-1)
        st.write(predicted_prob)
        if(int(ind)<5):
            st.write(labels[int(ind)])
        else:
            st.write(labels[int(ind)-1])
