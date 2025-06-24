from sklearn.metrics import accuracy_score, precision_recall_fscore_support

classes = {
    0: 'World',
    1: 'Sports',
    2: 'Business',
    3: 'Sci/Tech'
}

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def id2label(class_id):
    # assert class_id <= 3 and class_id >= 0, f'Unexpected class id {class_id}'
    return classes[class_id]