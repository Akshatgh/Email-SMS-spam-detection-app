import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from streamlit_lottie import st_lottie
import requests
import os

# Define a custom directory to store NLTK data
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Set the NLTK data path
nltk.data.path.append(nltk_data_dir)

# Download necessary NLTK resources to the custom directory
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)



def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_spam = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_4kx2q32n.json")
st_lottie(lottie_spam, height=150)

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
    }
    </style>
    """,
    unsafe_allow_html=True
)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.sidebar.title("About")
st.sidebar.info(
    "This Email/SMS Spam Detection app is designed to classify text messages as Spam or Not Spam with high accuracy. It uses Natural Language Processing (NLP) techniques for text cleaning, tokenization, and feature extraction, powered by the NLTK library. The app applies ensemble learning methods—combining the strengths of multiple machine learning models—to improve prediction performance and reduce false positives.\n\n"
    "Whether it's a suspicious email or a promotional SMS, this tool helps users make informed decisions quickly. The app features an intuitive Streamlit interface, allowing users to enter any message and receive instant classification results."
)

st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>📧 Email/SMS Spam Classifier</h1>",
    
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: #555;'>Detect spam messages instantly using Machine Learning!</p>",
    unsafe_allow_html=True
)

# Add a little animation (spinner) when predicting
input_sms = st.text_area("✉️ Enter the message", height=150)

if st.button('Predict'):
    with st.spinner('Analyzing message...'):
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]

    # 4. Display with color
    if result == 1:
        st.markdown(
            "<div style='background-color:#FF4B4B; padding:20px; border-radius:10px; text-align:center;'>"
            "<h2 style='color:white;'>🚫 Spam</h2>"
            "</div>",
            unsafe_allow_html=True
        )
        st.balloons()
    else:
        st.markdown(
            "<div style='background-color:#4BB543; padding:20px; border-radius:10px; text-align:center;'>"
            "<h2 style='color:white;'>✅ Not Spam</h2>"
            "</div>",
            unsafe_allow_html=True
        )
        st.snow()

st.markdown(
    "<hr><p style='text-align:center; color: #888;'>Made with ❤️ by Akshat</p>",
    unsafe_allow_html=True
)

