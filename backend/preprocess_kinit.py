import os
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import re

nltk.download('punkt')
nltk.download('punkt_tab')

BASE_DIR = "../datasets/KinitData"
OUTPUT_DIR = "../datasets/KINIT/processed"
INPUT_FILE = os.path.join(BASE_DIR, "fact_checking_articles.csv")

CHUNK_SIZE_CHARS = 1500

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text):
    sentences = sent_tokenize(text)
    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) <= CHUNK_SIZE_CHARS:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current:
        chunks.append(current.strip())
    return chunks

def map_label(rating):
    if not isinstance(rating, str):
        return None
    rating = rating.lower()

    credible_terms = ["true", "mostly true"]
    noncredible_terms = [
        "false", "mostly false", "mixture", "mixed", "half true",
        "unverified", "unknown", "misleading", "inaccurate"
    ]

    for t in credible_terms:
        if t in rating:
            return 1

    for t in noncredible_terms:
        if t in rating:
            return 0

    return None

def preprocess_kinit():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading:", INPUT_FILE)
    df = pd.read_csv(INPUT_FILE)

    if "raw_description" in df.columns:
        df["text"] = df["raw_description"].astype(str)
    elif "description" in df.columns:
        df["text"] = df["description"].astype(str)
    else:
        print("No text field found")
        return

    df["label"] = df["rating"].apply(map_label)
    df = df.dropna(subset=["label"])

    all_chunks = []

    for _, row in df.iterrows():
        cleaned = clean_text(row["text"])
        chunks = chunk_text(cleaned)

        for c in chunks:
            all_chunks.append({
                "text": c,
                "label": int(row["label"])
            })

    out_df = pd.DataFrame(all_chunks)

    out_path = os.path.join(OUTPUT_DIR, "kinit_data.csv")
    out_df.to_csv(out_path, index=False)

    print(f"Saved KINIT: {out_path}")
    print(f"Total samples: {len(out_df)}")

if __name__ == "__main__":
    preprocess_kinit()
