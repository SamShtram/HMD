import os
import re
import pandas as pd

# --------------- CLEANER ---------------
def clean_text(text):
    # Normalize spaces
    text = re.sub(r"\s+", " ", text)
    
    # Remove HTML artifacts
    text = re.sub(r"<.*?>", " ", text)

    # Remove scripts/styles
    text = re.sub(r"(?s)<script.*?</script>", " ", text)
    text = re.sub(r"(?s)<style.*?</style>", " ", text)

    # Remove cookie/legal repeated text
    text = re.sub(r"(cookie|privacy|advertisement|subscribe).*?", "", text, flags=re.I)

    # Remove escape characters
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"\\t", " ", text)

    return text.strip()

# --------------- CHUNKER ---------------
def chunk_text(text, max_words=256, overlap=50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

    return chunks

# --------------- MERGE CREDIBLE + FAKE ---------------
def process_folder(folder_path, label):
    rows = []
    files = os.listdir(folder_path)

    for file_name in files:
        full_path = os.path.join(folder_path, file_name)

        # Skip non-text files
        if not full_path.endswith(".txt"):
            continue

        # Read file
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read()
        except:
            continue

        # Clean
        cleaned = clean_text(text)

        # Chunk
        chunks = chunk_text(cleaned)

        # Add rows
        for chunk in chunks:
            rows.append({"text": chunk, "label": label})

    return rows

# ---------- PATHS (Update if needed) ----------
credible_folder = "datasets/credible_raw"
fake_folder      = "datasets/fake_raw"

# ---------- BUILD DATAFRAME ----------
rows = []

print("Processing credible articles...")
rows += process_folder(credible_folder, label=1)

print("Processing fake articles...")
rows += process_folder(fake_folder, label=0)

df = pd.DataFrame(rows)
df.to_csv("datasets/cleaned_final_dataset.csv", index=False)

print("DONE!")
print("Total samples:", len(df))
print(df.head())
