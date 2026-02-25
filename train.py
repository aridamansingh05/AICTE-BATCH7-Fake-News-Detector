import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os


# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Load dataset
df_fake = pd.read_csv(r"C:\Users\Aridaman Singh\OneDrive\Desktop\Live Fake News Detector\Fake.csv")
df_true = pd.read_csv(r"C:\Users\Aridaman Singh\OneDrive\Desktop\Live Fake News Detector\True.csv")

# Add label column
df_fake["label"] = "FAKE"
df_true["label"] = "REAL"

# Combine datasets
df = pd.concat([df_fake, df_true], axis=0)

# Combine title + text
df["content"] = df["title"] + " " + df["text"]

X = df["content"]
y = df["label"]

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

