from transformers import pipeline

MODEL_PATH = r"C:\Users\samms\HMD\backend\distilbert_model"

classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    local_files_only=True
)

text = "Vaccines cause autism."
result = classifier(text)

print(result)
