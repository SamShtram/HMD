import os
import json
import pandas as pd

DATASETS_DIR = "../datasets"

def load_scifact():
    scifact_path = os.path.join(DATASETS_DIR, "SciFact/claims_train.jsonl")
    if not os.path.exists(scifact_path):
        return pd.DataFrame(columns=["text", "label"])

    rows = []
    with open(scifact_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("claim", "")
            label = 0
            if obj.get("evidence"):
                label = 1
            rows.append({"text": text, "label": label})

    df = pd.DataFrame(rows)
    return df


def load_fakehealth():
    base = os.path.join(DATASETS_DIR, "FakeHealth/content")

    text_list = []
    label_list = []

    folders = ["HealthRelease", "HealthStory"]

    for folder in folders:
        folder_path = os.path.join(base, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if not file.endswith(".json"):
                continue

            full_path = os.path.join(folder_path, file)
            with open(full_path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            text = obj.get("text", "")
            label = 1 if obj.get("rating", 0) < 3 else 0

            text_list.append(text)
            label_list.append(label)

    return pd.DataFrame({"text": text_list, "label": label_list})


def load_kinit(kinit_dir):
    claims_path = os.path.join(kinit_dir, "claims.csv")
    articles_path = os.path.join(kinit_dir, "articles.csv")
    fc_articles_path = os.path.join(kinit_dir, "fact_checking_articles.csv")

    # Load CSVs
    claims = pd.read_csv(claims_path)
    articles = pd.read_csv(articles_path)
    fc_articles = pd.read_csv(fc_articles_path)

    # Merge claims with article text
    merged = claims.merge(articles, on="article_id", how="left")

    # Merge fact-checking labels (TRUE/FALSE)
    merged = merged.merge(fc_articles[["claim_id", "verdict"]], on="claim_id", how="left")

    # Map verdict to binary labels
    label_map = {
        "TRUE": 1,
        "PARTLY_TRUE": 1,
        "FALSE": 0,
        "UNSUPPORTED": 0,
        "MISLEADING": 0
    }

    merged["label"] = merged["verdict"].map(label_map)

    # Use article text + claim text combined
    merged["text"] = merged["claim"] + " | " + merged["article_text"]

    # Drop empty rows
    merged = merged[["text", "label"]].dropna()

    return merged


def main():
    print("\n=== Merging SciFact + FakeHealth + KINIT ===")

    # Paths
    fakehealth_dir = os.path.join(DATASETS_DIR, "FakeHealth/content")
    healthrelease_dir = os.path.join(fakehealth_dir, "HealthRelease")
    healthstory_dir = os.path.join(fakehealth_dir, "HealthStory")

    scifact_dir = os.path.join(DATASETS_DIR, "SciFact")

    # IMPORTANT: your folder is named KinitData
    kinit_dir = os.path.join(DATASETS_DIR, "KinitData")

    # Load datasets
    scifact = load_scifact(scifact_dir)
    print(f"SciFact: {scifact.shape}")

    fake_hr = load_fakehealth_folder(healthrelease_dir)
    fake_hs = load_fakehealth_folder(healthstory_dir)
    fake = pd.concat([fake_hr, fake_hs], ignore_index=True)
    print(f"FakeHealth: {fake.shape}")

    # Load KINIT (new data source)
    kinit = load_kinit(kinit_dir)
    print(f"KINIT: {kinit.shape}")

    # Merge all datasets together
    full_df = pd.concat([scifact, fake, kinit], ignore_index=True)
    print(f"\nFinal merged dataset shape: {full_df.shape}")

    # Save final CSV
    output_path = os.path.join(DATASETS_DIR, "merged_final_dataset.csv")
    full_df.to_csv(output_path, index=False)
    print(f"Final merged dataset saved to: {output_path}")



if __name__ == "__main__":
    main()
