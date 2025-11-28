import pandas as pd
import re
import joblib
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
nltk.download("omw-1.4")

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+", " url ", text)
        text = re.sub(r"\S+@\S+", " email ", text)
        text = re.sub(r"\d+", " number ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = " ".join(self.lemmatizer.lemmatize(w) for w in text.split())
        return text

    def transform(self, X, y=None):
        return [self.clean_text(x) for x in X]

    def fit(self, X, y=None):
        return self


def load_data():
    df = pd.read_csv("../datasets/cleaned_health_dataset.csv")

    df = df[df["text"].str.len() > 40]

    return df


def build_pipeline():
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        stop_words="english",
        sublinear_tf=True
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3
    )

    features = FeatureUnion([
        ("word_features", word_vectorizer),
        ("char_features", char_vectorizer)
    ])

    model = LinearSVC(class_weight="balanced")

    pipeline = Pipeline([
        ("cleaner", TextCleaner()),
        ("features", features),
        ("classifier", model)
    ])

    return pipeline


def train():
    df = load_data()
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(pipeline, "model_pipeline.joblib")
    print("Model pipeline saved.")


if __name__ == "__main__":
    train()
