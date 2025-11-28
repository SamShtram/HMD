import json
import csv

INPUT_FILES = [
    "../datasets/SciFact/claims_train.jsonl",
    "../datasets/SciFact/claims_dev.jsonl",
    "../datasets/SciFact/claims_test.jsonl",
]

OUTPUT_FILE = "../datasets/SciFact/processed/scifact_data.csv"


def preprocess_scifact():
    rows = []

    for file in INPUT_FILES:
        print(f"Loading {file}...")
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)

                claim_text = item.get("claim", "").strip()
                rating = item.get("label", "").upper()

                # SciFact uses: SUPPORTS, REFUTES, NOT_ENOUGH_INFO
                if rating == "SUPPORTS":
                    label = 1
                elif rating == "REFUTES":
                    label = 0
                else:
                    # Skip NOT_ENOUGH_INFO (NEI)
                    continue

                if claim_text:
                    rows.append({
                        "text": claim_text,
                        "label": label
                    })

    print(f"Total SciFact samples kept: {len(rows)}")

    # Write final CSV
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved SciFact dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    preprocess_scifact()
