import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from utils import compute_metrics

def main():

    # Load dataset & model
    data = load_dataset("fancyzhx/ag_news")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)

    def tokenize_function(examples, max_length=128):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

    # encoding
    encoded_data = data.map(tokenize_function, batched=True)
    encoded_data = encoded_data.rename_column("label", "labels")  # rename label to labels
    encoded_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # training 
    training_args = TrainingArguments(
        output_dir="./results",
        logging_dir="./results/logs",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        save_strategy="epoch",
        metric_for_best_model="accuracy",
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_data["train"],
        eval_dataset=encoded_data["test"],
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()

    # save model
    trainer.save_model("./model")
    tokenizer.save_pretrained("./model")

if __name__ == "__main__":
    main()
