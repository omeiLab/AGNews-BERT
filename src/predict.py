from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt
from utils import id2label

model_path = './model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
    logits = outputs.logits.squeeze()
    probs = torch.softmax(logits, dim = -1)

    labels = [id2label(i) for i in range(4)]
    prob_values = probs.tolist()

    fig, ax = plt.subplots(figsize = (7, 3))
    ax.barh(labels, prob_values, color='skyblue', height=0.35)
    ax.set_xlim(0, 1)
    for i, v in enumerate(prob_values):
        ax.text(v+0.01, i, f'{v:.2f}', va = 'center')
    plt.title(f"It's a {id2label(predicted_class_id)} news")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()

    return fig
