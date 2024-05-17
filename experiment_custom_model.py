import os
import sys
import wandb
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
from datetime import datetime
from datasets import Dataset
from torch.nn import Parameter, CrossEntropyLoss
import torch.nn.functional as F
from dotenv import load_dotenv
from omegaconf import OmegaConf
import huggingface_hub as hf_hub
from huggingface_hub import HfApi
from huggingface_hub import login
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


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float())
        sum_embeddings      = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask            = input_mask_expanded.sum(1)
        sum_mask            = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings     = sum_embeddings / sum_mask
        return mean_embeddings


class AESModel(DebertaV2PreTrainedModel):
    def __init__(self, user_config = None, model_config=None):
        super().__init__(config)
        self.deberta    = DebertaV2Model(model_config)
        self.num_labels = model_config.num_labels

        for i in range(0, user_config.num_freez_layers, 1):
            for n,p in self.deberta.encoder.layer[i].named_parameters():
                p.requires_grad = False

        self.pooler     = MeanPooling()
        self.classifier = nn.Linear(model_config.hidden_size, self.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids            = None,
        attention_mask       = None,
        labels               = None,
        output_hidden_states = None,
    ):
        outputs = self.deberta(
            input_ids,
            attention_mask       = attention_mask,
            output_hidden_states = output_hidden_states,
        )

        last_hidden_state = outputs[0]
        pooled_output     = self.pooler(last_hidden_state, attention_mask)
        logits            = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss     = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states
        )

def get_model(config):

    tokenizer    = AutoTokenizer.from_pretrained(config.model_id)
    model_config = AutoConfig.from_pretrained(config.model_id, num_labels=config.num_labels)
    model        = AESModel(user_config = config,model_config=model_config)

    return tokenizer, model

def prepare_dataset(config, dataset_df):

    dataset_df['score'] = dataset_df['score'] - 1
    dataset_df          = dataset_df.rename(columns={'score':'label'})
    dataset_obj         = Dataset.from_pandas(dataset_df)

    return dataset_obj

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    qwk                 = cohen_kappa_score(labels, predictions.argmax(-1), weights="quadratic")
    results             = {"qwk": qwk}
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
    
    api.upload_file(
        path_or_fileobj="experiment.py",
        path_in_repo=path_in_repo,
        repo_id=config.HUGGINGFACE_REPO,
        repo_type="model",
        )
    api.upload_file(
        path_or_fileobj="config.yaml",
        path_in_repo=path_in_repo,
        repo_id=config.HUGGINGFACE_REPO,
        repo_type="model",
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

    start_time = datetime.now()

    try:
        from discordwebhook import Discord
        discord        = Discord(url=os.environ["DISCORD_WEBHOOK"])
        notify_discord = True
    except Exception as e:
        print(f"will not able to log to discord cause of error : {e}")
        notify_discord = False

    if notify_discord:
        discord.post(
            content=f"🚀 Starting experiment {config.experiment_name} at time : {start_time}"
        )

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
    finish_str = f"🎉 Experiment {cfg.experiment_name} completed at time: {end_time}. Total time taken : {(end_time-start_time)/60} minutes."
    if notify_discord:
        discord.post(content=finish_str)
    
    print(f"Total time taken by experiment {(end_time-start_time)/60} minutes.")
    print(f"This is the end.....")

if __name__ == "__main__":
    config_file_path = sys.argv.pop(1)
    cfg              = OmegaConf.merge(OmegaConf.load(config_file_path), OmegaConf.from_cli())
    main(cfg)
