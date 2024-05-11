import sys
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


tokenizer      = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
tokenized_imdb = imdb.map(preprocess_function, batched=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


if __name__ == "__main__":

    fold  = sys.argv[1]
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
    )


    training_args = TrainingArguments(
        output_dir                  = "my_awesome_model",
        learning_rate               = 2e-5,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size  = 16,
        num_train_epochs            = 2,
        weight_decay                = 0.01,
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        push_to_hub                 = True,
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = tokenized_imdb["train"],
        eval_dataset    = tokenized_imdb["test"],
        tokenizer       = tokenizer,
        data_collator   = data_collator,
        compute_metrics = compute_metrics,
    )

    trainer.train()

trainer.push_to_hub()