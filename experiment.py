import os
import sys
import wandb
import numpy as np
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from omegaconf import OmegaConf
import huggingface_hub as hf_hub
from huggingface_hub import HfApi
from huggingface_hub import login
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoConfig, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed


load_dotenv()

def prepare_dataset(config, dataset_df):

    dataset_df['score'] = dataset_df['score'] - 1
    dataset_df          = dataset_df.rename(columns={'score':'label'})
    dataset_obj         = Dataset.from_pandas(dataset_df)

    return dataset_obj

def get_model(config):

    tokenizer       = AutoTokenizer.from_pretrained(config.model_id)
    model_config    = AutoConfig.from_pretrained(config.model_id, num_labels=config.num_labels)
    model           = AutoModelForSequenceClassification.from_pretrained(config.model_id, config=model_config)

    return tokenizer, model

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    qwk                 = cohen_kappa_score(labels, predictions.argmax(-1), weights="quadratic")
    results             = {"qwk": qwk}
    return results

def tokenize_function(example,tokenizer,truncation,max_length):
    return tokenizer(example["full_text"],truncation=truncation, max_length=max_length,  return_token_type_ids=False)

def push_to_huggingface(config, out_dir):
    
    login(token=os.environ["HF_TOKEN"], write_permission=True)  

    repo_id = os.environ["HF_TOKEN"] + '/' + config.experiment_name
    api     = HfApi()
    
    print(f"Uploading files to huggingface repo...")

    repo_url     = hf_hub.create_repo(repo_id, exist_ok=True, private=True)
    path_in_repo = f"fold_{fold}"
    api.upload_folder(
        folder_path=out_dir, repo_id=repo_id, path_in_repo=path_in_repo
    )

    print(f"All output folder is push to huggingface repo for experiment : {config.experiment_name}")

def inference(config, trainer, eval_dataset, eval_df, out_dir):

    logits, _, _       = trainer.predict(eval_dataset)
    predictions        = logits.argmax(-1) + 1
    eval_df["pred"]   = predictions

    logits_df          = pd.DataFrame(logits, columns=[f"pred_{i}" for i in range(1, 7)])
    result_df          = pd.concat([eval_df, logits_df], axis=1)

    file_path          = out_dir + '/' +f"fold_{config.fold}_oof.csv"
    result_df.to_csv(file_path, index=False)

    print(f"OOF is saved at : {file_path}")

def main(config):

    out_dir = os.path.join(config.output_dir,f"fold_{config.fold}")
    os.makedirs(out_dir, exist_ok = True)

    set_seed(config.seed)

    if config.wandb_log:
        wandb.init(
                        project = config.wandb_project_name,
                        group   = config.experiment_name,
                        name    = f"fold_{config.fold}",
                        notes   = config.notes,
                        config  = OmegaConf.to_container(config, resolve=True)
                   )

    dataset_df        = pd.read_csv(os.path.join(config.data_dir,config.training_filename))

    train_df          = dataset_df[dataset_df["fold"] != config.fold]
    eval_df           = dataset_df[dataset_df["fold"] == config.fold]

    train_dataset     = prepare_dataset(config, train_df)
    eval_dataset      = prepare_dataset(config, eval_df)

    tokenizer, model  = get_model(config)

    train_dataset     = train_dataset.map(tokenize_function, batched=True, fn_kwargs={'tokenizer':tokenizer,'truncation':config.truncation,'max_length':config.max_length})
    eval_dataset      = eval_dataset.map(tokenize_function, batched=True, fn_kwargs={'tokenizer':tokenizer,'truncation':config.truncation,'max_length':config.max_length})

    data_collator     = DataCollatorWithPadding(tokenizer=tokenizer)

    args              = TrainingArguments(output_dir=out_dir, **config.training_args)

    trainer           = Trainer (
                                    model           = model,
                                    args            = args,
                                    train_dataset   = train_dataset,
                                    eval_dataset    = eval_dataset,
                                    data_collator   = data_collator,
                                    compute_metrics = compute_metrics,
                                )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    inference(config, trainer, eval_dataset, eval_df, out_dir)
    push_to_huggingface(config, out_dir)

    print(f"This is the end.....")



if __name__ == "__main__":
    config_file_path = sys.argv.pop(1)
    cfg              = OmegaConf.merge(OmegaConf.load(config_file_path), OmegaConf.from_cli())
    main(cfg)
