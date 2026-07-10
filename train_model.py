import pandas as pd
import nltk
import joblib
from preprocess import transform_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# Keep only useful columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

# Dataset information
print("Dataset Shape:", df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate values
print("\nDuplicate Messages:", df.duplicated().sum())

# Remove duplicate messages
df = df.drop_duplicates()

print("\nShape After Removing Duplicates:", df.shape)

# Convert labels into numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
# Create a new column with cleaned text
df['transformed_message'] = df['message'].apply(transform_text)

print("\nTransformed Messages:")
print(df[['message', 'transformed_message']].head())

print("\nConverted Labels:")
print(df.head())
# Create TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=3000)

# Convert text into numerical vectors
X = tfidf.fit_transform(df['transformed_message']).toarray()

# Target variable
y = df['label']

print("\nShape of Feature Matrix:", X.shape)
print("Shape of Target:", y.shape)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
# Create the model
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Display results
print("\nModel Performance")
print("----------------------------")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Save the trained model
joblib.dump(model, "model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(tfidf, "vectorizer.pkl")

print("\nModel and Vectorizer saved successfully!")