import os
import sys
import wandb
import warnings
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime
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

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


load_dotenv()


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float() #expanding the attention mask to match the dimensions of last_hidden_state. unsqueeze(-1) adds a new dimension to the attention mask tensor
        sum_embeddings      = torch.sum(last_hidden_state * input_mask_expanded, 1)                 #This line calculates the sum of the element-wise multiplication of last_hidden_state and input_mask_expanded along the second dimension (axis 1). 
        sum_mask            = input_mask_expanded.sum(1)                                            #This line calculates the sum of input_mask_expanded along the second dimension, resulting in a tensor containing the count of non-padding tokens for each input sequence.
        sum_mask            = torch.clamp(sum_mask, min=1e-9)                                       #This line clamps the values of sum_mask to ensure that they are not too small, preventing division by zero errors. 
        mean_embeddings     = sum_embeddings / sum_mask
        return mean_embeddings

class CustomModel(nn.Module):
    def __init__(self, config=None, pretrained=False):
        super().__init__()
        self.model_config = None
        if pretrained:
            self.model_config = AutoConfig.from_pretrained(config.model_id, num_labels=config.num_labels)
            model             = AutoModelForSequenceClassification.from_pretrained(config.model_id, config=self.model_config)
            self.pool         = MeanPooling()
            self.fc           = nn.Linear(self.model_config.hidden_size, config.num_labels)
            self._init_weights(self.fc)
        else:
            print("Loading model for inference.....")
            
    def _init_weights(self, module):
        """
        This method initializes weights for different types of layers. The type of layers
        supported are nn.Linear, nn.Embedding and nn.LayerNorm.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        """
        This method makes a forward pass through the model, get the last hidden state (embedding)
        and pass it through the MeanPooling layer.
        """
        outputs            = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature            = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        """
        This method makes a forward pass through the model, the MeanPooling layer and finally
        then through the Linear layer to get a regression value.
        """
        feature = self.feature(inputs)
        output  = self.fc(feature)
        return output


def get_model(config):

    tokenizer  = AutoTokenizer.from_pretrained(config.model_id)
    model      = CustomModel(config=config, pretrained=True)

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
