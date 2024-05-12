import os
import sys
import wandb
import numpy as np
import pandas as pd
from datasets import Dataset
from omegaconf import OmegaConf
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoConfig, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed
from dotenv import load_dotenv

load_dotenv()

def prepare_dataset(config):
    dataset_df  = pd.read_csv(os.path.join(config.data_dir,config.training_filename))
    dataset_df['score'] = dataset_df['score'] - 1
    dataset     = Dataset.from_pandas(dataset_df)
    return dataset

def get_model(config):

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model_config    = AutoConfig.from_pretrained(config.model_id,num_labels=config.num_labels)
    model     = AutoModelForSequenceClassification.from_pretrained(config.model_id,config=model_config)

    return tokenizer, model


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    qwk                 = cohen_kappa_score(labels, predictions.argmax(-1), weights="quadratic")
    results             = {"qwk": qwk}
    return results

def tokenize_function(example,tokenizer,truncation,max_length):
    return tokenizer(example["full_text"],truncation=truncation, max_length=max_length,  return_token_type_ids=False)

def main(config):

    out_dir = os.path.join(config.output_dir,f"fold_{config.fold}")
    os.makedirs(out_dir, exist_ok = True)

    set_seed(config.seed)

    if config.wandb_log:
        wandb.init(
            project=config.wandb_project_name,
            group=config.experiment_name,
            name=f"{config.experiment_name}_fold_{config.fold}",
            notes=config.notes,
            config=OmegaConf.to_container(config, resolve=True),
        )

    
    dataset           = prepare_dataset(config)

    train_dataset = dataset.filter(lambda example: example["fold"] != config.fold)
    eval_dataset  = dataset.filter(lambda example: example["fold"] == config.fold)

    tokenizer, model  = get_model(config)

    train_dataset = train_dataset.map(tokenize_function, batched=True, fn_kwargs={'tokenizer':tokenizer,'truncation':config.truncation,'max_length':config.max_length})
    eval_dataset  = eval_dataset.map(tokenize_function, batched=True, fn_kwargs={'tokenizer':tokenizer,'truncation':config.truncation,'max_length':config.max_length})

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(output_dir=out_dir, **config.training_args)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()



if __name__ == "__main__":
    config_file_path = sys.argv.pop(1)
    cfg              = OmegaConf.merge(OmegaConf.load(config_file_path), OmegaConf.from_cli())
    main(cfg)
