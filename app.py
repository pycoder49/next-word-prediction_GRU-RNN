from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import numpy as np
import pickle

# loading the model
model = load_model('model/next_word_lstm.keras')

# loading the tokenizer
with open('model/tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

# function to predict the nextw word
def predict_next_word(model, tokenizer, text, max_length):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_length:
      token_list = token_list[-(max_length-1):]

    token_list = pad_sequences([token_list], maxlen=max_length-1, padding="pre")
    predicted_word = model.predict(token_list, verbose=2)
    predicted_word_index = np.argmax(predicted_word, axis=1)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return None

# streamlit app
st.title("Next Word Prediction using GRU-RNN")
input_text = st.text_input("Enter a sequence of words:", "Predict the next ")
st.button("Predict Next Word")

if input_text:
    max_sequence_len = model.input_shape[1]
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"Predicted Next Word: {next_word}")