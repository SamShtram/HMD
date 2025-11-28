import pandas as pd
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    df = pd.read_csv("../datasets/merged_health_dataset.csv")
    
    # Clean text
    df["text"] = df["text"].astype(str).apply(clean_text)

    
    # Remove extremely short samples (hurts accuracy)
    df = df[df["text"].str.len() > 60]
    print("Removed samples shorter than 60 characters.")
  

    # Drop duplicate rows
    df = df.drop_duplicates(subset=["text"])

    # Reset index
    df = df.reset_index(drop=True)

    df.to_csv("../datasets/cleaned_health_dataset.csv", index=False)
    print("Final shape:", df.shape)


if __name__ == "__main__":
    main()
