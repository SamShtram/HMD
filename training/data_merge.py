import os
import json
import pandas as pd

BASE = "../datasets"

def load_scifact(scifact_dir):
    """
    Loads SciFact claims from claims_train/dev/test.jsonl.
    Normalizes labels:
        SUPPORTED -> 0 (real)
        REFUTED -> 1 (misinformation)
        NOINFO -> dropped
    """
    rows = []
    files = ["claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl"]

    for file in files:
        path = os.path.join(scifact_dir, file)
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                claim = item.get("claim", "").strip()
                label = item.get("label", "")

                if label == "SUPPORTED":
                    y = 0
                elif label == "REFUTED":
                    y = 1
                else:
                    continue

                if len(claim) > 0:
                    rows.append({"text": claim, "label": y})

    df = pd.DataFrame(rows)
    print("Loaded SciFact:", df.shape)
    return df


def load_fakehealth_reviews(path):
    """
    Loads review JSON containing rating values.
    Returns a dict mapping url/canonical_url to numeric rating.
    """
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ratings = {}

    for entry in data:
        source_link = entry.get("source_link", "").strip()
        link = entry.get("link", "").strip()
        rating = entry.get("rating", None)

        if rating is None:
            continue

        key_candidates = [source_link, link]
        for key in key_candidates:
            if key is not None and len(key) > 0:
                ratings[key] = rating

    return ratings


def load_fakehealth_articles(folder_path, review_ratings):
    """
    Loads FakeHealth article JSON files and merges them with review ratings.
    Matches canonical_link or url against review source_link/link.
    """
    rows = []

    for file in os.listdir(folder_path):
        if not file.endswith(".json"):
            continue

        fp = os.path.join(folder_path, file)

        with open(fp, "r", encoding="utf-8") as f:
            doc = json.load(f)

        text = doc.get("text", "")
        if isinstance(text, list):
            text = " ".join(text)
        text = text.strip()

        if len(text) == 0:
            continue

        url = doc.get("url", "").strip()
        canonical = doc.get("canonical_link", "").strip()

        rating = None

        if canonical in review_ratings:
            rating = review_ratings[canonical]
        elif url in review_ratings:
            rating = review_ratings[url]
        else:
            continue

        if rating in [-1, None]:
            continue
        elif rating in [0, 1]:
            y = 0
        elif rating in [2, 3]:
            y = 1
        else:
            continue

        rows.append({"text": text, "label": y})

    df = pd.DataFrame(rows)
    print("Loaded FakeHealth from", folder_path, ":", df.shape)
    return df


def main():
    scifact_dir = os.path.join(BASE, "SciFact")

    reviews_release_path = os.path.join(BASE, "FakeHealth", "reviews", "HealthRelease.json")
    reviews_story_path = os.path.join(BASE, "FakeHealth", "reviews", "HealthStory.json")

    release_ratings = load_fakehealth_reviews(reviews_release_path)
    story_ratings = load_fakehealth_reviews(reviews_story_path)

    release_articles_path = os.path.join(BASE, "FakeHealth", "content", "HealthRelease")
    story_articles_path = os.path.join(BASE, "FakeHealth", "content", "HealthStory")

    df_scifact = load_scifact(scifact_dir)

    df_release = load_fakehealth_articles(release_articles_path, release_ratings)
    df_story = load_fakehealth_articles(story_articles_path, story_ratings)

    merged = pd.concat([df_scifact, df_release, df_story], ignore_index=True)
    print("Final merged dataset shape:", merged.shape)

    out_path = os.path.join(BASE, "merged_health_dataset.csv")
    merged.to_csv(out_path, index=False)

    print("Merged dataset saved to:", out_path)


if __name__ == "__main__":
    main()
