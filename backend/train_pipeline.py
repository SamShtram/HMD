import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset


# ============================================================
# 1. Load merged dataset
# ============================================================

DATA_PATH = "../datasets/final_merged_dataset.csv"

print(f"[LOAD] Loading merged dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

df = df[["text", "label"]]   # ensure only these two columns
df = df.dropna()

print(df.head())
print(df["label"].value_counts())


# ============================================================
# 2. Train/Validation split
# ============================================================

train_df, val_df = train_test_split(
    df, test_size=0.15, random_state=42, stratify=df["label"]
)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)


# ============================================================
# 3. Tokenizer
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

print("[TOKENIZE] Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["text", "__index_level_0__"])
val_dataset = val_dataset.remove_columns(["text", "__index_level_0__"])

train_dataset.set_format("torch")
val_dataset.set_format("torch")


# ============================================================
# 4. Compute class weights (handles imbalance)
# ============================================================

def compute_class_weights(dataset):
    labels = np.array(dataset["label"])
    class_counts = np.bincount(labels)
    total = labels.shape[0]

    weights = total / (len(class_counts) * class_counts)
    weights = torch.tensor(weights, dtype=torch.float32)

    print(f"[INFO] Class counts: {class_counts}")
    print(f"[INFO] Class weights: {weights}")

    return weights

class_weights = compute_class_weights(train_dataset)


# ============================================================
# 5. WeightedTrainer subclass
# ============================================================

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        """
        HuggingFace Trainer now sometimes passes extra arguments such as 
        num_items_in_batch or others. We absorb them via *args/**kwargs so
        compute_loss never breaks.
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ============================================================
# 6. Metrics for evaluation
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )

    try:
        probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auc
    }


# ============================================================
# 7. Load model
# ============================================================

print("[MODEL] Loading DistilBERT model...")
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)


# ============================================================
# 8. TrainingArguments
# ============================================================

training_args = TrainingArguments(
    output_dir="./model_output",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=True,   # mixed precision if GPU supports it
    report_to="none"
)


# ============================================================
# 9. Trainer instance
# ============================================================

trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# ============================================================
# 10. Train model
# ============================================================

print("\n[TRAIN] Starting training...\n")
trainer.train()


# ============================================================
# 11. Evaluate
# ============================================================

print("\n[EVAL] Final evaluation...\n")
metrics = trainer.evaluate()
print(metrics)


# ============================================================
# 12. Save model + tokenizer
# ============================================================

trainer.save_model("./model_output/best_model")
tokenizer.save_pretrained("./model_output/best_model")

print("\n[SAVED] Model + tokenizer saved to ./model_output/best_model\n")
