import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

DATA_PATH = "../datasets/cleaned_health_dataset.csv"
MODEL_OUT = "../models/distilbert_health_model"

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
    return df

class HealthDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def main():
    df = load_data()

    split = int(len(df) * 0.8)
    train_df = df[:split]
    test_df = df[split:]

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_dataset = HealthDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
    test_dataset = HealthDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="./bert_output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()
    trainer.evaluate()

    model.save_pretrained(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)

    print(f"DistilBERT model saved to: {MODEL_OUT}")

if __name__ == "__main__":
    main()
