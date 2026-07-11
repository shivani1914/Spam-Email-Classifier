# 📧 Spam Email Classifier
## 🌐 Live Demo

👉 **Try the application here:**

**https://spam-email-classifier-w4tc6taan2nywm4pw9mwxp.streamlit.app/**
---
A Machine Learning project that classifies email or SMS messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) and the Multinomial Naive Bayes algorithm.

## Features

- Text preprocessing using NLTK
- TF-IDF Vectorization
- Multinomial Naive Bayes Classifier
- Streamlit Web Application
- Spam/Not Spam prediction

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NLTK
- Streamlit
- Joblib

## Dataset

SMS Spam Collection Dataset

## Project Structure

```
Spam-Email-Classifier/
│
├── dataset/
│   └── spam.csv
├── app.py
├── train_model.py
├── preprocess.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md
```

## Accuracy

- Accuracy: **97.2%**
- Precision: **99.15%**

## Run the Project

Install dependencies

```
pip install -r requirements.txt
```

Run the application

```
python -m streamlit run app.py
```

## Author

Shivani Mukhare