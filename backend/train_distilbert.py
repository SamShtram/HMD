import os
import pandas as pd
from datasets import Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load cleaned dataset
df = pd.read_csv("../datasets/cleaned_final_dataset.csv")

# Ensure text is string
df["text"] = df["text"].astype(str)

# Drop rows where text is too short or invalid
df = df[df["text"].str.strip().str.len() > 5]


# HuggingFace expects column name `label`
df = df.rename(columns={"labels": "label", "Labels": "label"})

# Convert to HF Dataset
dataset = Dataset.from_pandas(df)

# Train / test split
dataset_split = dataset.train_test_split(test_size=0.15)
train_ds = dataset_split["train"]
test_ds = dataset_split["test"]

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    texts = batch["text"]

    # ensure everything is string
    texts = [str(t) if not isinstance(t, str) else t for t in texts]

    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=384
    )


train_ds = train_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize, batched=True)

# Set format for PyTorch
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
).to(device)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

# Correct argument names for Transformers 4.57.3
training_args = TrainingArguments(
    output_dir="./distilbert_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=6,
    weight_decay=0.01,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
results = trainer.evaluate()
print("Evaluation:", results)

pred = trainer.predict(test_ds)
y_true = pred.label_ids
y_pred = np.argmax(pred.predictions, axis=1)

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Not Credible", "Credible"]))

trainer.save_model("./distilbert_model")
tokenizer.save_pretrained("./distilbert_model")
print("Model saved.")
