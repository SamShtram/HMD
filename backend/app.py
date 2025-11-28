import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests
from bs4 import BeautifulSoup
import re
import numpy as np

app = FastAPI()

MODEL_PATH = "./model_output/best_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

LABELS = {0: "Not Credible", 1: "Credible"}


# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -----------------------------
# SIMPLE SCRAPER (NO NEWSPAPER)
# -----------------------------
def extract_text_from_url(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, timeout=10, headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {e}")

    if res.status_code != 200:
        raise HTTPException(status_code=400, detail="Website returned non-200 status")

    soup = BeautifulSoup(res.text, "html.parser")

    # Remove scripts, styles, nav, footer, etc.
    for tag in soup(["script", "style", "header", "footer", "nav", "aside", "form"]):
        tag.extract()

    # Extract all text from paragraphs
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = " ".join(paragraphs)

    cleaned = clean_text(text)

    if len(cleaned.split()) < 30:
        raise HTTPException(status_code=400, detail="Not enough readable text extracted")

    return cleaned


# -----------------------------
# REQUEST MODEL
# -----------------------------
class URLInput(BaseModel):
    url: str


# -----------------------------
# PREDICT ENDPOINT
# -----------------------------
@app.post("/predict_url")
def predict_url(input: URLInput):
    url = input.url

    text = extract_text_from_url(url)

    # Tokenize
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred = int(np.argmax(probs))
    confidence = float(probs[pred])

    return {
        "url": url,
        "label": LABELS[pred],
        "confidence": confidence,
        "raw_scores": probs.tolist(),
        "preview": text[:400]
    }


# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
