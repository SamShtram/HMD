import pandas as pd
import re

input_path = "../datasets/merged_final_dataset.csv"
output_path = "../datasets/cleaned_final_dataset.csv"

df = pd.read_csv(input_path)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r'[^A-Za-z0-9.,!?;:\'"()/%+\s-]', ' ', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["text"] = df["text"].astype(str).apply(clean_text)


df.to_csv(output_path, index=False)
print("Cleaned saved to:", output_path)
print("Final shape:", df.shape)
