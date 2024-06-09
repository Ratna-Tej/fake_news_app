import streamlit as st 
import pandas as pd
import re
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

vectorization=pickle.load(open('vectorization.pkl','rb'))
LR=pickle.load(open('LR.pkl','rb'))
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)    
    return text

def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    return pred_LR



# Set page configuration
st.set_page_config(page_title="Fake News Classification App", page_icon="📰", layout="centered")

# CSS for custom styling

st.markdown("""
    <style>
    body {
        background-color: #ffffff; /* Set background to grey */
    }
    .title {
        font-size: 3em;
        font-weight: bold;
        color: #0099ff;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 1.5em;
        color: #666;
         font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }
    .stTextArea textarea {
        background-color: white !important; /* Set text box background to white */
        color: black !important;
        border: 2px solid #4CAF50 !important;
        border-radius: 8px !important;
        padding: 10px !important;
        cursor: text !important; /* Ensure cursor is visible and text-like */
    }
    .stTextArea textarea:focus {
        border-color: #45a049 !important; /* Change border color on focus */
        box-shadow: 0 0 5px rgba(69, 160, 73, 0.5) !important; /* Add focus effect */
        outline: none !important; /* Remove default outline */
    }
    .stButton>button {
        background-color: green !important;
        color: white !important;
        border: none;
        padding: 10px 20px !important;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px !important;
        margin: 4px 2px !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    .stButton>button:hover {
        background-color: #45a049 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Main title

st.markdown('<div class="title">Fake News Classification App 📰</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Input the news content below</div>', unsafe_allow_html=True)

# Input text area
sentence = st.text_area("Enter your news content here", "", height=200, key="text_area", help="Type the news article you want to classify.")

# Prediction button
predict_btt = st.button("Predict", key="predict_button")

if predict_btt:
    prediction_class = manual_testing(sentence)
    if prediction_class == [0]:
        st.error('This news is classified as Fake.')
    elif prediction_class == [1]:
        st.success('This news is classified as True.')
    else:
        st.error('Unable to classify the news content.')

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #888; font-weight: bold">© 2024 Fake News Classification App</div>', unsafe_allow_html=True)
