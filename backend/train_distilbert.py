import pandas as pd
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load cleaned dataset
df = pd.read_csv("../datasets/cleaned_health_dataset.csv")

# HF datasets require column name 'label'
df = df.rename(columns={"label": "labels"})

# Convert to HF Dataset
dataset = Dataset.from_pandas(df)

# Train / test split
dataset = dataset.train_test_split(test_size=0.15)
train_ds = dataset["train"]
test_ds = dataset["test"]

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
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
).to(device)

# Training args compatible with current Transformers
training_args = TrainingArguments(
    output_dir="./distilbert_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=3,
    warmup_steps=100,
    weight_decay=0.01,

    load_best_model_at_end=True,

    fp16=True,   # Mixed precision (only works on GPU)
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

trainer.train()

# Save model + tokenizer
save_dir = "./model/distilbert"
trainer.model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("\nTraining complete. Model saved to:", save_dir)
