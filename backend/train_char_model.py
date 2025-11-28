import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    df = pd.read_csv("../datasets/cleaned_health_dataset.csv")

    # Drop duplicates again just to be safe
    df = df.drop_duplicates(subset=["text"])

    # Filter for length > 60 (improves signal/noise)
    df = df[df["text"].str.len() > 60]

    return df


def build_pipeline():
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 6),
        min_df=3,
        sublinear_tf=True
    )

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("vectorizer", vectorizer),
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

    joblib.dump(pipeline, "char_model.joblib")
    print("Char-level model saved.")


if __name__ == "__main__":
    train()

