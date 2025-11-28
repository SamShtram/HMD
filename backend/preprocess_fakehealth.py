import json
import os
import glob
import pandas as pd
import nltk
import re

nltk.download("punkt")

DATA_ROOT = "C:/Users/samms/HMD/datasets/FakeHealth"
CONTENT_DIR = f"{DATA_ROOT}/content"
REVIEW_DIR = f"{DATA_ROOT}/reviews"
OUT_DIR = f"{DATA_ROOT}/processed"
os.makedirs(OUT_DIR, exist_ok=True)

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return text.strip()

def load_content():
    content_map = {}

    for folder in ["HealthRelease", "HealthStory"]:
        path = f"{CONTENT_DIR}/{folder}/*.json"

        for fp in glob.glob(path):
            filename = os.path.basename(fp)
            news_id = filename.replace(".json", "")  # Extract ID from filename

            with open(fp, "r", encoding="utf8") as f:
                data = json.load(f)

            text = clean_text(data.get("text", ""))

            if text:
                content_map[news_id] = text

    print("Loaded content items:", len(content_map))
    return content_map

def load_reviews():
    reviews = []
    for fp in glob.glob(f"{REVIEW_DIR}/*.json"):
        with open(fp, "r", encoding="utf8") as f:
            data = json.load(f)

        if isinstance(data, list):
            reviews.extend(data)
        else:
            reviews.append(data)

    print("Loaded review items:", len(reviews))
    return reviews

def preprocess():
    content = load_content()
    reviews = load_reviews()

    rows = []
    missing = 0

    for r in reviews:
        news_id = r.get("news_id")
        if not news_id:
            continue

        rating = r.get("rating")  # 1 = credible, 2 = not credible
        label = 1 if rating == 1 else 0

        if news_id not in content:
            missing += 1
            continue

        text = content[news_id]

        if len(text) < 200:
            continue

        rows.append({
            "text": text,
            "label": label,
            "news_id": news_id
        })

    print("Matched items:", len(rows))
    print("Missing content:", missing)

    df = pd.DataFrame(rows)

    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
    test_df.to_csv(f"{OUT_DIR}/test.csv", index=False)

    print("Saved:")
    print(" Train:", f"{OUT_DIR}/train.csv")
    print(" Test:", f"{OUT_DIR}/test.csv")

if __name__ == "__main__":
    preprocess()
