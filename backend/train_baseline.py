import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("../datasets/cleaned_health_dataset.csv")

X = df["text"].astype(str)
y = df["label"].astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Vectorizer
vectorizer = TfidfVectorizer(
    max_features=25000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=500)

model.fit(X_train_vec, y_train)

# Evaluation
preds = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:")
print(classification_report(y_test, preds))

# Save artifacts
joblib.dump(model, "../backend/model.joblib")
joblib.dump(vectorizer, "../backend/vectorizer.joblib")

print("\nModel and vectorizer saved.")
