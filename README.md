# 📧 Email/SMS Spam Classifier

This is a machine learning web app built using **Streamlit**, which can classify messages as **Spam** or **Not Spam**.

## 🚀 Features

- 🔍 Real-time classification of input text as spam or ham.
- 📊 Preprocessing using NLTK (stemming, tokenization, stopword removal).
- 🧠 Machine Learning model with TF-IDF Vectorization.
- 🎨 Clean and interactive UI with Lottie animations using Streamlit.

## 🛠️ How it Works

1. User inputs a message.
2. The message is cleaned, tokenized, and stemmed using NLTK.
3. It is transformed using a pre-trained TF-IDF vectorizer.
4. A machine learning model classifies it as Spam/Not Spam.

## 💾 Files in this Project

- `app.py` – Main Streamlit application
- `model.pkl` – Pre-trained ML model
- `vectorizer.pkl` – TF-IDF vectorizer
- `requirements.txt` – All dependencies
- `Dataset/` – Contains original dataset
- `.gitignore` – Specifies files to ignore in version control

## 📚 Tech Stack

- **Frontend**: Streamlit
- **ML Libraries**: scikit-learn, NLTK, pickle
- **Model**: TF-IDF + Naive Bayes 
- **Deployment**: Streamlit Cloud 


## 🧪 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Akshatgh/Email-SMS-spam-detection-app.git
cd your-repo-name

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # on Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

🙋‍♂️ Author
Made with ❤️ by Akshat