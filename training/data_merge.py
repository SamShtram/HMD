import os
import json
import pandas as pd

# -----------------------------
# SciFact Loader
# -----------------------------
def load_scifact(scifact_dir):
    combined = []
    files = ["claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl"]

    for fname in files:
        path = os.path.join(scifact_dir, fname)
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                claim = item.get("claim", "")
                cited = item.get("cited_doc_ids", [])

                # Credibility mapping
                label = 1 if len(cited) > 0 else 0

                combined.append({"text": claim, "label": label})

    return pd.DataFrame(combined)


# -----------------------------
# FakeHealth Loader
# -----------------------------
def load_fakehealth(fakehealth_dir):
    import os
    import json
    import pandas as pd

    entries = []

    # Correct folder names inside FakeHealth/content/
    folders = ["HealthRelease", "HealthStory"]

    # Path to the real directory containing them
    content_dir = os.path.join(fakehealth_dir, "content")

    for folder in folders:
        json_dir = os.path.join(content_dir, folder)

        if not os.path.exists(json_dir):
            print("Missing folder:", json_dir)
            continue

        for fname in os.listdir(json_dir):
            if fname.endswith(".json"):
                full_path = os.path.join(json_dir, fname)

                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                title = data.get("title", "")
                description = data.get("description", "")
                text = f"{title} {description}".strip()

                # FakeHealth rating field → credible vs not credible
                rating = str(data.get("rating", "")).lower()
                credible = ["true", "mostly true", "mixture"]
                label = 1 if rating in credible else 0

                entries.append({"text": text, "label": label})

    df = pd.DataFrame(entries)
    return df



# -----------------------------
# KINIT Dataset Loader
# -----------------------------
def load_kinit(kinit_dir):
    import pandas as pd
    import os

    claims_path = os.path.join(kinit_dir, "claims.csv")
    if not os.path.exists(claims_path):
        print("KINIT claims.csv not found!")
        return pd.DataFrame(columns=["text", "label"])

    df = pd.read_csv(claims_path)

    # Use `statement` as the text
    df["text"] = df["statement"].astype(str)

    # Convert rating -> binary credibility label
    df["label"] = df["rating"].apply(
        lambda x: 1 if str(x).lower() in ["true", "mostly true", "mixture"] else 0
    )

    return df[["text", "label"]]



# -----------------------------
# MAIN MERGE LOGIC
# -----------------------------
def main():
    base = "../datasets"

    print("Loading FakeHealth...")
    fakehealth = load_fakehealth(os.path.join(base, "FakeHealth"))
    print("FakeHealth:", fakehealth.shape)

    print("Loading SciFact...")
    scifact = load_scifact(os.path.join(base, "SciFact"))
    print("SciFact:", scifact.shape)

    print("Loading KINIT...")
    kinit = load_kinit(os.path.join(base, "KinitData"))
    print("KINIT:", kinit.shape)

    merged = pd.concat([fakehealth, scifact, kinit], ignore_index=True)

    out_path = os.path.join(base, "merged_final_dataset.csv")
    merged.to_csv(out_path, index=False)

    print("Done. Final shape:", merged.shape)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
