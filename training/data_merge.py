import os
import json
import pandas as pd

BASE = "../datasets"


def load_scifact(scifact_dir):
    """
    Loads SciFact claims and extracts labels from nested evidence.
    Rules:
        SUPPORT -> 0 (real)
        CONTRADICT -> 1 (misinformation)
        NOT_ENOUGH_INFO or empty -> dropped
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
                evidence = item.get("evidence", {})

                if len(claim) == 0:
                    continue

                label_found = None

                # Extract labels from nested evidence structure
                for evid_list in evidence.values():
                    for evid in evid_list:
                        lbl = evid.get("label", "")
                        if lbl == "SUPPORT":
                            label_found = 0
                        elif lbl == "CONTRADICT":
                            label_found = 1

                if label_found is None:
                    continue

                rows.append({
                    "text": claim,
                    "label": label_found
                })

    df = pd.DataFrame(rows)
    print("Loaded SciFact:", df.shape)
    return df


def load_fakehealth_reviews(path):
    """
    Loads FakeHealth review JSON containing rating values.
    Returns a dictionary mapping url or canonical url to rating.
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

        # Map both keys to allow flexible matching
        key_candidates = [source_link, link]
        for key in key_candidates:
            if key is not None and len(key) > 0:
                ratings[key] = rating

    return ratings


def load_fakehealth_articles(folder_path, review_ratings):
    """
    Loads FakeHealth articles and matches them to ratings using
    canonical_link or url.
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

        # Try to match the article to a rating
        if canonical in review_ratings:
            rating = review_ratings[canonical]
        elif url in review_ratings:
            rating = review_ratings[url]
        else:
            continue

        # Normalize ratings: 0/1 real, 2/3 misinformation, -1 ignore
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
    print("=== Merging SciFact + FakeHealth Datasets ===")

    # SciFact directory
    scifact_dir = os.path.join(BASE, "SciFact")

    # FakeHealth review JSONs
    reviews_release_path = os.path.join(BASE, "FakeHealth", "reviews", "HealthRelease.json")
    reviews_story_path = os.path.join(BASE, "FakeHealth", "reviews", "HealthStory.json")

    release_ratings = load_fakehealth_reviews(reviews_release_path)
    story_ratings = load_fakehealth_reviews(reviews_story_path)

    # FakeHealth article directories
    release_articles_path = os.path.join(BASE, "FakeHealth", "content", "HealthRelease")
    story_articles_path = os.path.join(BASE, "FakeHealth", "content", "HealthStory")

    # Load datasets
    df_scifact = load_scifact(scifact_dir)
    df_release = load_fakehealth_articles(release_articles_path, release_ratings)
    df_story = load_fakehealth_articles(story_articles_path, story_ratings)

    # Merge all datasets
    merged = pd.concat([df_scifact, df_release, df_story], ignore_index=True)
    print("Final merged dataset shape:", merged.shape)

    # Save result
    out_path = os.path.join(BASE, "merged_health_dataset.csv")
    merged.to_csv(out_path, index=False)

    print("Merged dataset saved to:", out_path)


if __name__ == "__main__":
    main()
