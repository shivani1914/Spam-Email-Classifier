import streamlit as st
import joblib
from preprocess import transform_text

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📧 Spam Email Classifier")

input_sms = st.text_area("Enter your message")

if st.button("Predict"):

    transformed_sms = transform_text(input_sms)

    vector_input = vectorizer.transform([transformed_sms])

    result = model.predict(vector_input)

    if result[0] == 1:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Not Spam")