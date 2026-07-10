import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Load stopwords only once
stop_words = set(stopwords.words('english'))

def transform_text(text):
    # Convert to lowercase
    text = text.lower()

    # Split sentence into words
    text = nltk.word_tokenize(text)

    y = []

    # Keep only letters and numbers
    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    # Remove stopwords
    for word in text:
        if word not in stop_words:
            y.append(word)

    text = y[:]
    y.clear()

    # Apply stemming
    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)