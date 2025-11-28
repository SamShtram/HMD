import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "./distilbert_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

def predict(text):
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=384, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]

    label_id = torch.argmax(probs).item()
    confidence = probs[label_id].item()

    return ("Credible" if label_id == 1 else "Not Credible"), confidence

if __name__ == "__main__":
    text = "This is a sample health claim to classify."
    label, conf = predict(text)
    print(label, conf)
