from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./model",
    tokenizer="./model",
    return_all_scores=True
)

while True:
    text = input("\nEnter text: ")
    result = classifier(text)
    print(result)
