import os
import json
import pandas as pd

DATASETS_DIR = "../datasets"



# FAKEHEALTH LOADER

def load_fakehealth(folder):
    health_release = os.path.join(folder, "content", "HealthRelease")
    health_story = os.path.join(folder, "content", "HealthStory")

    rows = []

    def load_folder(path):
        for fname in os.listdir(path):
            if fname.endswith(".json"):
                fpath = os.path.join(path, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    text = data.get("text", "")
                    label = 1 if data.get("rating", 0) >= 3 else 0
                    rows.append([text, label])
                except:
                    continue

    if os.path.exists(health_release):
        load_folder(health_release)
    if os.path.exists(health_story):
        load_folder(health_story)

    return pd.DataFrame(rows, columns=["text", "label"])



# SCIFACT LOADER

def load_scifact(scifact_dir):
    claims_train = os.path.join(scifact_dir, "claims_train.jsonl")
    claims_dev = os.path.join(scifact_dir, "claims_dev.jsonl")

    rows = []

    # choose whichever exists
    path = claims_train if os.path.exists(claims_train) else claims_dev
    if not os.path.exists(path):
        print("SciFact not found.")
        return pd.DataFrame(columns=["text", "label"])

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = item.get("claim", "")
            cited = item.get("cited_doc_ids", [])
            label = 1 if len(cited) > 0 else 0
            rows.append([text, label])

    return pd.DataFrame(rows, columns=["text", "label"])



# KINIT LOADER (medical misinformation CSV dataset)

def load_kinit(kinit_dir):
    claims_path = os.path.join(kinit_dir, "claims.csv")
    fc_path = os.path.join(kinit_dir, "fact_checking_articles.csv")

    if not os.path.exists(claims_path):
        print("KINIT not found.")
        return pd.DataFrame(columns=["text", "label"])

    df_claims = pd.read_csv(claims_path)
    df_fc = pd.read_csv(fc_path) if os.path.exists(fc_path) else pd.DataFrame(columns=["id"])

    # text source: claim body
    df_claims["text"] = df_claims["claim"]
    df_claims["label"] = df_claims["verified"].apply(lambda x: 1 if x == "true" else 0)

    # optional fact-checking articles may override labels (not required)
    merged = df_claims[["text", "label"]]

    return merged.dropna()



# MAIN MERGE PIPELINE

def main():
    print("\n=== Merging SciFact + FakeHealth + KINIT ===\n")

    # ----- FAKEHEALTH -----
    fakehealth_dir = os.path.join(DATASETS_DIR, "FakeHealth")
    fakehealth = load_fakehealth(fakehealth_dir)
    print(f"FakeHealth: {fakehealth.shape}")

    # ----- SCIFACT -----
    scifact_dir = os.path.join(DATASETS_DIR, "SciFact")
    scifact = load_scifact(scifact_dir)
    print(f"SciFact: {scifact.shape}")

    # ----- KINIT -----
    kinit_dir = os.path.join(DATASETS_DIR, "KinitData")
    kinit = load_kinit(kinit_dir)
    print(f"KINIT: {kinit.shape}")

    # merge
    merged = pd.concat([fakehealth, scifact, kinit], ignore_index=True)

    # shuffle
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = os.path.join(DATASETS_DIR, "merged_final_dataset.csv")
    merged.to_csv(out_path, index=False)

    print(f"\nFinal merged dataset saved to: {out_path}")
    print(f"Shape: {merged.shape}")


if __name__ == "__main__":
    main()
