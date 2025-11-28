import pandas as pd
from datasets import Dataset
import torch
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
df = pd.read_csv("../datasets/cleaned_health_dataset.csv")

# HuggingFace expects column name `label`
df = df.rename(columns={"labels": "label"})

# Convert to HF Dataset
dataset = Dataset.from_pandas(df)

# Train / test split
dataset_split = dataset.train_test_split(test_size=0.15)
train_ds = dataset_split["train"]
test_ds = dataset_split["test"]

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Set format for PyTorch
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
).to(device)

# Correct argument names for Transformers 4.57.3
training_args = TrainingArguments(
    output_dir="./distilbert_model",
    eval_strategy="epoch",          # <-- NEW NAME
    save_strategy="epoch",          # <-- NEW NAME
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

trainer.train()
eval_results = trainer.evaluate()
print(eval_results)

model.save_pretrained("./distilbert_model")
tokenizer.save_pretrained("./distilbert_model")
print("DistilBERT model saved.")
