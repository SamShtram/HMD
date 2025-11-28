import os
import json
import pandas as pd

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
                label = 1 if cited else 0
                combined.append({"text": claim, "label": label})

    return pd.DataFrame(combined)

def load_fakehealth(fakehealth_dir):
    texts = []
    files = ["HealthRelease", "HealthStory"]

    for folder in files:
        json_dir = os.path.join(fakehealth_dir, folder)
        if not os.path.exists(json_dir):
            continue

        for fname in os.listdir(json_dir):
            if fname.endswith(".json"):
                path = os.path.join(json_dir, fname)
                data = json.load(open(path, "r", encoding="utf-8"))
                text = data.get("title", "") + " " + data.get("description", "")
                label = 1 if data.get("rating", 0) >= 3 else 0
                texts.append({"text": text, "label": label})

    return pd.DataFrame(texts)

def load_kinit(kinit_dir):
    claims_path = os.path.join(kinit_dir, "claims.csv")
    articles_path = os.path.join(kinit_dir, "articles.csv")
    factcheck_path = os.path.join(kinit_dir, "fact_checking_articles.csv")

    df_claims = pd.read_csv(claims_path, low_memory=False)
    df_articles = pd.read_csv(articles_path, low_memory=False)
    df_fact = pd.read_csv(factcheck_path, low_memory=False)

    df_claims["id"] = df_claims["id"].astype(str)
    df_articles["id"] = df_articles["id"].astype(str)
    df_fact["id"] = df_fact["id"].astype(str)

    merged_claim_article = df_claims.merge(
        df_articles[["id", "body"]],
        how="left",
        left_on="id",
        right_on="id"
    )

    merged_claim_article_fact = merged_claim_article.merge(
        df_fact[["id", "claim"]],
        how="left",
        left_on="id",
        right_on="id"
    )

    merged_claim_article_fact["text"] = merged_claim_article_fact.apply(
        lambda row: (
            str(row["statement"]) if pd.notna(row["statement"]) else ""
        ) + " " + (
            str(row["body"]) if pd.notna(row["body"]) else ""
        ) + " " + (
            str(row["claim"]) if pd.notna(row["claim"]) else ""
        ),
        axis=1
    )

    merged_claim_article_fact["label"] = merged_claim_article_fact["rating"].apply(
        lambda x: 1 if str(x).lower() == "true" else 0
    )

    return merged_claim_article_fact[["text", "label"]]

def main():
    base = "../datasets"

    fakehealth_dir = os.path.join(base, "FakeHealth", "content")
    scifact_dir = os.path.join(base, "Scifact")
    kinit_dir = os.path.join(base, "KinitData")

    print("FakeHealth:")
    fakehealth = load_fakehealth(fakehealth_dir)
    print(fakehealth.shape)

    print("SciFact:")
    scifact = load_scifact(scifact_dir)
    print(scifact.shape)

    print("KINIT:")
    kinit = load_kinit(kinit_dir)
    print(kinit.shape)

    final_df = pd.concat([fakehealth, scifact, kinit], ignore_index=True)
    out_path = os.path.join(base, "merged_final_dataset.csv")
    final_df.to_csv(out_path, index=False)

    print("Done. Final shape:", final_df.shape)

if __name__ == "__main__":
    main()
