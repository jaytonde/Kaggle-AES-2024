import os
import sys
import wandb
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
from datasets import Dataset
from torch.nn import Parameter, CrossEntropyLoss
import torch.nn.functional as F
from dotenv import load_dotenv
from omegaconf import OmegaConf
import huggingface_hub as hf_hub
from huggingface_hub import HfApi
from huggingface_hub import login
from tokenizers import AddedToken
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoModelForSequenceClassification

from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    DebertaV2Model,
    DebertaV2PreTrainedModel,
    ProgressCallback,
    Trainer,
    TrainingArguments,
    set_seed
)

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

load_dotenv()


def get_model(config):

    tokenizer    = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.add_tokens([AddedToken("\n", normalized=False)])
    tokenizer.add_tokens([AddedToken(" "*2, normalized=False)])

    model_config = AutoConfig.from_pretrained(config.model_id)
    
    model_config.attention_probs_dropout_prob = 0.0 
    model_config.hidden_dropout_prob          = 0.0 
    model_config.num_labels                   = 1 

    model           = AutoModelForSequenceClassification.from_pretrained(config.model_id, config=model_config)

    return tokenizer, model

def prepare_dataset(config, dataset_df):

    dataset_df['score'] = dataset_df['score'] - 1
    dataset_df["score"] = dataset_df["score"].astype('float32')
    dataset_df          = dataset_df.rename(columns={'score':'label'})
    dataset_obj         = Dataset.from_pandas(dataset_df)

    return dataset_obj

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    qwk                 = cohen_kappa_score(labels, predictions.clip(0,5).round(0), weights='quadratic')
    results             = {'qwk': qwk}
    return results

def tokenize_function(example,tokenizer,truncation,max_length):
    return tokenizer(example["full_text"],truncation=truncation, max_length=max_length,  return_token_type_ids=False)

def push_to_huggingface(config, out_dir):
    
    login(token=os.environ["HF_TOKEN"], write_permission=True)  

    repo_id = os.environ["HF_USERNAME"] + '/' + config.experiment_name
    api     = HfApi()
    
    print(f"Uploading files to huggingface repo...")

    if config.full_fit:
        repo_url     = hf_hub.create_repo(repo_id, exist_ok=True, private=True)
        path_in_repo = f"full_fit"
        api.upload_folder(
            folder_path=out_dir, repo_id=repo_id, path_in_repo=path_in_repo
        )
    else:
        repo_url     = hf_hub.create_repo(repo_id, exist_ok=True, private=True)
        path_in_repo = f"fold_{config.fold}"
        api.upload_folder(
            folder_path=out_dir, repo_id=repo_id, path_in_repo=path_in_repo
        )
    
    print(f"Current working dir : {os.getcwd()}")
    root_dir = f"../"
    print(f"training file path : {os.path.join(root_dir, config.train_code_file)}")
    api.upload_file(
        path_or_fileobj=os.path.join(root_dir, config.train_code_file),
        path_in_repo="experiment.py",
        repo_id=repo_id,
        repo_type="model",
        )
    api.upload_file(
        path_or_fileobj=os.path.join(root_dir ,"config.yaml"),
        path_in_repo="config.yaml",
        repo_id=repo_id,
        repo_type="model",
        )

    print(f"All output folder is push to huggingface repo for experiment : {config.experiment_name}")


def inference(config, trainer, eval_dataset, eval_df, out_dir):

    try:
        print(f"Starting with inference on validation data")
        predictions_thre   = trainer.predict(eval_dataset).predictions
        predictions        = predictions_thre.round(0) + 1
        eval_df["pred"]    = predictions

        logits_df              = pd.DataFrame()
        logits_df['pred_thre'] = predictions_thre
        result_df              = pd.concat([eval_df, logits_df], axis=1)

        file_path          = out_dir + '/' +f"fold_{config.fold}_oof.csv"
        result_df.to_csv(file_path, index=False)

        print(f"OOF is saved at : {file_path}")
        
        cm                 = confusion_matrix(eval_df['score'], eval_df["pred"], labels=[x for x in range(1,7)])
        draw_cm            = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[x for x in range(1,7)])
        draw_cm.plot()
        plt.show()

        print(f"Completed with inference on validation data")
    except Exception as e:
        print(f"Error while doing inference : {e}")

def main(config):

    start_time = datetime.now()

    if config.debug:
        print(f"Debugging mode is on.....")
        
    if config.full_fit:
        print(f"Running experiment in full_fit mode.....")
        out_dir = os.path.join(config.output_dir,f"full_fit")
    else:
        print(f"Running experiment in folding mode.....")
        out_dir = os.path.join(config.output_dir,f"fold_{config.fold}")

    os.makedirs(out_dir, exist_ok = True)

    set_seed(config.seed)

    if config.wandb_log:
        name = ''
        if config.full_fit:
            name = "full_fit"
        else:
            name = f"fold_{config.fold}"

        wandb.init(
                        project = config.wandb_project_name,
                        group   = config.experiment_name,
                        name    = name,
                        notes   = config.notes,
                        config  = OmegaConf.to_container(config, resolve=True)
                )

    dataset_df        = pd.read_csv(os.path.join(config.data_dir,config.training_filename))

    if config.debug:
        train_df          = dataset_df[0:1000]
        eval_df           = dataset_df[1001:2050]
        train_dataset     = prepare_dataset(config, train_df)
        eval_dataset      = prepare_dataset(config, eval_df)
    else:    
        if config.full_fit:
            train_df          = dataset_df
            train_dataset     = prepare_dataset(config, train_df)
            eval_dataset      = None
        else:
            train_df          = dataset_df[dataset_df["fold"] != config.fold]
            eval_df           = dataset_df[dataset_df["fold"] == config.fold]
            train_dataset     = prepare_dataset(config, train_df)
            eval_dataset      = prepare_dataset(config, eval_df)

    tokenizer, model  = get_model(config)

    train_dataset     = train_dataset.map(tokenize_function, batched=True, fn_kwargs={'tokenizer':tokenizer,'truncation':config.truncation,'max_length':config.max_length})

    if not config.full_fit:
        eval_dataset      = eval_dataset.map(tokenize_function, batched=True, fn_kwargs={'tokenizer':tokenizer,'truncation':config.truncation,'max_length':config.max_length})

    data_collator     = DataCollatorWithPadding(tokenizer=tokenizer)
    args              = TrainingArguments(output_dir=out_dir, **config.training_args)

    if config.full_fit:
        trainer           = Trainer (
                                        model           = model,
                                        args            = args,
                                        train_dataset   = train_dataset,
                                        data_collator   = data_collator,
                                        compute_metrics = compute_metrics,
                                    )
    else:
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

    if config.full_fit:
        inference(config, trainer, train_dataset, train_df, out_dir)
    else:
        inference(config, trainer, eval_dataset, eval_df, out_dir)

    push_to_huggingface(config, out_dir)

    end_time = datetime.now()
    
    print(f"Total time taken by experiment {(end_time-start_time)/60} minutes.")
    print(f"This is the end.....")


if __name__ == "__main__":
    config_file_path = sys.argv.pop(1)
    cfg              = OmegaConf.merge(OmegaConf.load(config_file_path), OmegaConf.from_cli())
    main(cfg)
