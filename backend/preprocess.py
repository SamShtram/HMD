import pandas as pd
import re
import os

INPUT_PATH = "../datasets/merged_health_dataset.csv"
OUTPUT_PATH = "../datasets/cleaned_health_dataset.csv"

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError("Merged dataset not found.")

    df = pd.read_csv(INPUT_PATH)

    df["text"] = df["text"].apply(clean_text)

    df = df[df["text"].str.len() > 10]

    df.to_csv(OUTPUT_PATH, index=False)

    print("Cleaning complete. Saved:", OUTPUT_PATH)
    print("Final shape:", df.shape)


if __name__ == "__main__":
    main()
