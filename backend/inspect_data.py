import pandas as pd

df = pd.read_csv("../datasets/cleaned_health_dataset.csv")

print("Dataset Inspection")
print("------------------")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nLabel Distribution:")
print(df["label"].value_counts())

print("\nSample Rows:")
print(df.head(5))
