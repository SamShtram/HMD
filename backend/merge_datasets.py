import pandas as pd
import os

# === INPUT FILES ===
FAKEHEALTH_TRAIN = "../datasets/FakeHealth/processed/train.csv"
FAKEHEALTH_TEST = "../datasets/FakeHealth/processed/test.csv"

CREDIBLE_FILE = "../datasets/Credible/processed/credible_data.csv"
KINIT_FILE = "../datasets/KINIT/processed/kinit_data.csv"

# === OUTPUT FILE ===
OUTPUT_FILE = "../datasets/final_merged_dataset.csv"


def load_if_exists(path):
    if not os.path.exists(path):
        print(f"[SKIP] File not found: {path}")
        return None
    print(f"[LOAD] {path}")
    return pd.read_csv(path)


def merge_datasets():
    dfs = []

    # Load mandatory datasets
    df_fake_train = load_if_exists(FAKEHEALTH_TRAIN)
    df_fake_test = load_if_exists(FAKEHEALTH_TEST)

    # Combine FakeHealth train+test
    df_fake = pd.concat([df_fake_train, df_fake_test], ignore_index=True)
    print(f"FakeHealth samples: {len(df_fake)}")

    dfs.append(df_fake)

    # Load Credible scraped articles
    df_cred = load_if_exists(CREDIBLE_FILE)
    if df_cred is not None:
        dfs.append(df_cred)
        print(f"Credible samples: {len(df_cred)}")

    # Load KINIT fact-checking articles
    df_kinit = load_if_exists(KINIT_FILE)
    if df_kinit is not None:
        dfs.append(df_kinit)
        print(f"KINIT samples: {len(df_kinit)}")

    # Merge all working datasets
    final_df = pd.concat(dfs, ignore_index=True)

    # Clean missing rows
    final_df = final_df.dropna(subset=["text", "label"])

    # Convert label to int
    final_df["label"] = final_df["label"].astype(int)

    print("Label distribution before shuffle:")
    print(final_df["label"].value_counts())

    # Shuffle dataset
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save final dataset
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n[OK] Final merged dataset saved to:")
    print(f"     {OUTPUT_FILE}")
    print(f"Total samples: {len(final_df)}")
    print("Label distribution after merge/shuffle:")
    print(final_df["label"].value_counts())


if __name__ == "__main__":
    merge_datasets()
