import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("Sentiment.csv")

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)  # remove links
    text = re.sub(r'[^a-z\s]', '', text)  # remove symbols
    return text

df['text'] = df['text'].apply(clean_text)

# Fix labels (important!)
df['sentiment'] = df['sentiment'].str.lower()

# Remove nulls only from required columns
df.dropna(subset=['text', 'sentiment'], inplace=True)

print(f"Loaded {len(df)} samples after cleaning")

X = df['text']
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Accuracy check
pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Improved model saved!")