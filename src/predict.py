from transformers import pipeline
import torch
from utils import id2label

classifier = pipeline("text-classification", model = "model")

def predict(text):
    result = classifier(text)[0]
    label = id2label(result['label'])
    score = round(result['score'] * 100, 2)

    return f"Predicted class: {label}\nConfidence = {score}%"
