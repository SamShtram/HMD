from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import trafilatura


app = FastAPI()

model_path = "./distilbert_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

class URLInput(BaseModel):
    url: str

def extract_text(url: str):
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded)
    return text

@app.post("/predict_url")
def predict_url(data: URLInput):
    text = extract_text(data.url)
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=1).cpu().detach().numpy()[0]

    pred_idx = int(probs.argmax())
    label = "Credible" if pred_idx == 1 else "Not Credible"
    confidence = float(probs[pred_idx])

    return {
        "label": label,
        "confidence": confidence,
        "extracted_chars": len(text)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
