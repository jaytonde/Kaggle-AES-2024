import sys
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


models_dict = {
    0 : "microsoft-debrta-v3-xsmall",
    1 : "microsoft-debrta-v3-small",
    2 : "microsoft-debrta-v3-base",
    3 : "microsoft-debrta-v3-large",
}


tokenizer      = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
tokenized_imdb = imdb.map(preprocess_function, batched=True)


class AESModel:
    def __init__(self):
        pass
    
    def forward():
        pass

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def prepare_dataset(self):
    dataset_dir = '/input/'
    dataset_df  = pd.read_csv(dataset_dir+'train.csv')
    dataset     = Dataset.from_pandas(dataset_df)
    return dataset

def get_training_args(): 
    training_args = TrainingArguments(
                                            output_dir                  = "output",
                                            learning_rate               = 2e-5,
                                            per_device_train_batch_size = 8,
                                            per_device_eval_batch_size  = 16,
                                            num_train_epochs            = 2,
                                            weight_decay                = 0.01,
                                            evaluation_strategy         = "epoch",
                                            save_strategy               = "epoch",
                                            load_best_model_at_end      = True,
                                            push_to_hub                 = False,
                                    )

    return training_args  

def get_trainer(model,training_args,dataloader,tokenizer,data_collator,compute_metrics):
trainer = Trainer(
                        model           = model,
                        args            = training_args,
                        train_dataset   = tokenized_imdb["train"],
                        eval_dataset    = tokenized_imdb["test"],
                        tokenizer       = tokenizer,
                        data_collator   = data_collator,
                        compute_metrics = compute_metrics,
                    )
return trainer

def save_model():
    pass



def main(config):

    model = AESModel()
    dataset       = utilities.prepare_dataset()
    training_args = utilities.get_training_args(model,training_args,dataset,tokenizer,data_collator,compute_metrics)
    trainer       = utilities.get_trainer()

    trainer.train()

    utilities.save_model()



if __name__ == "__main__":
    config_file_path = sys.argv.pop(1)
    cfg              = OmegaConf.merge(OmegaConf.load(config_file_path), OmegaConf.from_cli())
    main(cfg)
