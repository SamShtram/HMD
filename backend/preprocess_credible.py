import os
import re
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

# -------- CONFIG --------
RAW_CREDIBLE_DIR = r"../datasets/Credible/raw_txt"   # <-- your txt folder
OUTPUT_DIR = r"../datasets/Credible/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "credible_data.csv")
CHUNK_SIZE_CHARS = 1500  # approx ~512 tokens
# ------------------------


def clean_text(text):
    # remove URLs
    text = re.sub(r"http\S+", " ", text)

    # remove multiple newlines
    text = re.sub(r"\n+", "\n", text)

    # collapse extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def chunk_text(text, max_chars=1500):
    """splits text into chunks of max_chars"""
    chunks = []
    text = text.strip()

    if len(text) <= max_chars:
        return [text]

    sentences = sent_tokenize(text)
    current = ""

    for sent in sentences:
        if len(current) + len(sent) <= max_chars:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = sent

    if current:
        chunks.append(current.strip())

    return chunks


def preprocess_credible_articles():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    records = []

    print(f"Loading .txt files from: {RAW_CREDIBLE_DIR}")

    files = [f for f in os.listdir(RAW_CREDIBLE_DIR) if f.endswith(".txt")]

    print(f"Found {len(files)} files.")

    for fname in files:
        path = os.path.join(RAW_CREDIBLE_DIR, fname)

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
        except Exception as e:
            print(f"ERROR reading {fname}: {e}")
            continue

        cleaned = clean_text(raw)
        chunks = chunk_text(cleaned, CHUNK_SIZE_CHARS)

        for chunk in chunks:
            if len(chunk) > 50:  # avoid noise
                records.append({"text": chunk, "label": 1})

    print(f"Total chunks created: {len(records)}")

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved credible dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    preprocess_credible_articles()
