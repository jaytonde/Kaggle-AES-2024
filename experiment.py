import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoConfig, DataCollatorWithPadding
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

def prepare_dataset(config):
    dataset_df  = pd.read_csv(os.path.join(config.data_dir,config.training_filename))
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


def get_function(config):

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    config    = AutoConfig.from_pretrained(config.model_id,num_labels=config.num_labels)
    model     = AutoModelForSequenceClassification.from_pretrained(config.model_id,config=config)

    return tokenizer, model

def save_model():
    pass



def main(config):

    out_dir = os.path.join(config.output_dir,f"fold_{config.fold}")
    os.mkdir(out_dir, exist_ok = True)

    set_seed(cfg.seed)

    if cfg.wandb_log:
        cfg_dict = process_config_for_wandb(Config)
        cfg_dict.update(vars(cmd_args))
        wandb.init(
            project=cfg.wandb_project_name,
            group=cfg.experiment_name,
            name=f"{cfg.experiment_name}_fold_{fold}" if validate else f"{cfg.experiment_name}_full_fit",
            notes=cfg.notes,
            config=cfg_dict,
        )

    tokenizer, model  = get_model(config)
    dataset           = utilities.prepare_dataset(config)
    training_args     = utilities.get_training_args(tokenizer,dataset,model)
    trainer           = utilities.get_trainer()

    trainer.train()

    utilities.save_model()



if __name__ == "__main__":
    config_file_path = sys.argv.pop(1)
    cfg              = OmegaConf.merge(OmegaConf.load(config_file_path), OmegaConf.from_cli())
    main(cfg)
