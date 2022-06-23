"""
This script was converted from the CodeFill notebook in an attempt to get it to work.
Parts of the code were obtained from the CodeFill implementation reference at https://zenodo.org/record/5440779

I did not succeed in getting it to work but you still might like to read it.

Original notebook: https://github.com/saltudelft/codefill/blob/main/notebooks/CodeFill.ipynb
"""

# %% [markdown]
# Install the correct dependencies on HuggingFace transformer and tensorflow

# %%
# We won't need TensorFlow here
# %pip uninstall -y tensorflow
# Install `transformers` from master
# %pip install git+https://github.com/huggingface/transformers
# %pip list | grep -E 'transformers|tokenizers'
# %pip install nlp==0.2.0
# %pip install datasets
# %pip install git+https://github.com/huggingface/nlp
# %pip install sklearn

# transformers version at notebook update --- 2.11.0
# tokenizers version at notebook update --- 0.8.0rc1


# %%
import datetime
from io import BytesIO, StringIO
import os
import tokenize
import dis
import sys
import re
import keyword
import pandas as pd
import ast
import torch
import signal
from functools import wraps
from pathlib import Path
import json
from joblib import Parallel, delayed
from pathlib import Path
from transformers import AutoTokenizer, RobertaForQuestionAnswering,TextDataset,DataCollatorForLanguageModeling, trainer_utils, AutoModel
import transformers
import glob
import random
from cf_shared.utils import cf_glob, get_source_and_converted_paths, get_source_file_names_from_converted_folder, get_elapsed_us, print_elapsed_seconds, timed_cf_glob
from cf_shared.MultitaskModel import MultitaskModel
from cf_shared.convert import convert, convert_paths, get_successful_conversions

os.environ["CUDA_VISIBLE_DEVICES"]="1"

DATA_PATH = "/mnt/mturk/cf_sample_data/"

os.chdir(os.path.join(DATA_PATH, "script-environments/"))

RUN_NAME = "100k"
DEBUG_FILENAMES = False
PERFORM_CONVERSION = False
PERFORM_DATASET_COPY = False
DO_TRAIN = True
THREADS = 20
MAX_PATHS = 100_000
TIMES_JSON = "times.json"
CONVERTED_PATH = "./converted/"
PY_SOURCEFILES_LOCATION = os.path.join(DATA_PATH, "deduplicated_code_fill_pretrain/")

LOAD_PRETRAINED_MODEL = not DO_TRAIN

TRAINER_ARGS = transformers.TrainingArguments(
    output_dir=os.path.join(DATA_PATH, "checkpoints", RUN_NAME),
    overwrite_output_dir=DO_TRAIN,
    learning_rate=1e-5,
    do_train=DO_TRAIN,
    num_train_epochs=4,
    # Adjust batch size if this doesn"t fit on the Colab GPU
    per_device_train_batch_size=20,  
    fp16=True,
    save_steps=10000
)

run_path = "./" + RUN_NAME
if not os.path.exists(run_path):
    os.makedirs(run_path)

os.chdir(run_path)

if not os.path.exists(CONVERTED_PATH):
    os.makedirs(CONVERTED_PATH)

save_stdout = sys.stdout

WEIGHT_MATRIX = {
    'NUMBER' : [1.625, 1.25, 1.125],
    'NAME' : [1.625, 1.125, 1.5],
    'LOCAL_VARIABLE' : [1.625, 1.125, 1.5],
    'FUNCTION_NAME' : [1.625, 1.25, 1.5]
}

input_file = "./input_file.txt"
output_file = "./output_file.txt"
def reranking_layer(outputs, context, tokenizer):

  with open(input_file, 'w') as f:
    f.write(context)
  
  convert(file_path=input_file, output_file=output_file)
  with open(output_file, 'rb') as context:
    inputs = list(zip(tokenizer(input_file), tokenizer(output_file)))
    for item in inputs:
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(WEIGHT_MATRIX[item[1]]))

# %%

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE())

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# %%

tokenizer.train(files=["training_set"], trainer=trainer)

if PERFORM_CONVERSION:
    print("starting conversion")
    paths_input = timed_cf_glob(PY_SOURCEFILES_LOCATION, "*.py*")

    start_time = datetime.datetime.now()
    converted_paths_opt = convert_paths(paths_input, CONVERTED_PATH, times_json=TIMES_JSON, n_threads=20)
    print_elapsed_seconds(start_time, "converting files")
    paths, converted_paths = get_successful_conversions(converted_paths_opt)
else:
    print("skipping conversion")
    paths, converted_paths = get_source_and_converted_paths(CONVERTED_PATH, PY_SOURCEFILES_LOCATION)

print("converted file amount: " + str(len(converted_paths)))

if PERFORM_DATASET_COPY:
    start_time = datetime.datetime.now()
    print("starting the writing of source files")

    with open("./train.txt", "wb") as train_outfile:
        with open("./test.txt", "wb") as test_outfile:
            for f in paths:
                choice = random.random()
                with open(f, "rb") as infile:
                    if choice > 0.1:
                        train_outfile.write(infile.read())
                    else:
                        test_outfile.write(infile.read())

    # TODO: Parallelize these, they are independent and take a lot of time 
    # (provided the disks can do more I/O at a higher queue depth)
    print_elapsed_seconds(start_time, "the writing of source files")

    start_time = datetime.datetime.now()
    print("starting the writing of converted files")

    with open("./converted_train.txt", "wb") as train_outfile:
        with open("./converted_test.txt", "wb") as test_outfile:
            for f in converted_paths:
                choice = random.random()
                with open(f, "rb") as infile:
                    if choice > 0.1:
                        train_outfile.write(infile.read())
                    else:
                        test_outfile.write(infile.read())
    print_elapsed_seconds(start_time, "the writing of converted files")
    # print("the writing of converted files took: {:0.2f}s".format(get_elapsed_us(start_time)/1e6))

# %%
def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)
     
    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=128)   
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,test_dataset,data_collator

train_dataset,test_dataset,data_collator = load_dataset("./train.txt", "./test.txt",tokenizer)
converted_train_dataset, converted_test_dataset, converted_datacollator = load_dataset("./converted_train.txt", "./converted_test.txt",tokenizer)

train_dataset,test_dataset,data_collator = load_dataset("./train.txt", "./test_example.txt",tokenizer)
converted_train_dataset, converted_test_dataset, converted_datacollator = load_dataset("./converted_train.txt", "./converted_test_example.txt",tokenizer)

# raise Exception("stopping the script...")
# %%
tokenizer("for i in range(10)")["input_ids"]

# %%
import numpy as np
import torch
import torch.nn as nn
import transformers
import nlp
import logging
# from datasets import load_dataset
from transformers import TextDataset,DataCollatorForLanguageModeling


logging.basicConfig(level=logging.INFO)

dataset_dict = {
    "token": train_dataset,
    "token_type": converted_train_dataset,
    "line": train_dataset,
}

print(dataset_dict["token"])

print("loaded configs!")

# %%

max_length = 2040

def convert_to_token_features(batch):
    inputs = list(zip(batch['token'], batch['token_label']))
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = batch["token_label"]
    return features

def convert_to_type_features(batch):
    inputs = list(zip(batch['type'], batch['type_label']))
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = batch["type_lable"]
    return features



convert_func_dict = {
    "token": convert_to_token_features,
    "type": convert_to_type_features,
    "line": convert_to_token_features,
}

columns_dict = {
    "type": ['input_ids', 'attention_mask', 'labels'],
    "token": ['input_ids', 'attention_mask', 'labels'],
    "line": ['input_ids', 'attention_mask', 'labels'],
}

features_dict = {}
for task_name, dataset in dataset_dict.items():
    features_dict[task_name] = {}
    for phase, phase_dataset in dataset.items():
        features_dict[task_name][phase] = phase_dataset.map(
            convert_func_dict[task_name],
            batched=True,
            load_from_cache_file=False,
        )
        features_dict[task_name][phase].set_format(
            type="torch",
            columns=columns_dict[task_name],
        )

# %%

if LOAD_PRETRAINED_MODEL:
    multitask_model = AutoModel.from_pretrained('./pretrained_models/one-point-six-million')
else:
    model_name = "gpt2"
    multitask_model = MultitaskModel.create(
        model_name=model_name,
        model_type_dict={
            "token": transformers.AutoModelWithLMHead,
            "token_type": transformers.AutoModelWithLMHead,
            "line": transformers.AutoModelForSequenceClassification,
        },
        model_config_dict={
            "token": transformers.AutoConfig.from_pretrained(model_name),
            "token_type": transformers.AutoConfig.from_pretrained(model_name),
            "line": transformers.AutoConfig.from_pretrained(model_name),
        },
    )

# %%
# Check that we have a GPU
# !nvidia-smi
# Check that PyTorch sees it
import torch
print("cuda is available: " + str(torch.cuda.is_available()))

# %%
import dataclasses
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import DataCollatorForLanguageModeling, InputDataClass, DefaultDataCollator
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict
from transformers import Trainer
# from random import random
import random


class NLPDataCollator(DataCollatorForLanguageModeling):
    """
    Extending the existing DataCollator to work with NLP dataset batches
    """
    def collate_batch(self, features: List[Union[InputDataClass, Dict]]) -> Dict[str, torch.Tensor]:
        first = features[0]
        if isinstance(first, dict):
          # NLP data sets current works presents features as lists of dictionary
          # (one per example), so we  will adapt the collate_batch logic for that
          if "labels" in first and first["labels"] is not None:
              if first["labels"].dtype == torch.int64:
                  labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
              else:
                  labels = torch.tensor([f["labels"] for f in features], dtype=torch.float)
              batch = {"labels": labels}
          for k, v in first.items():
              if k != "labels" and v is not None and not isinstance(v, str):
                  batch[k] = torch.stack([f[k] for f in features])
          return batch
        else:
          # otherwise, revert to using the default collate_batch
          return DefaultDataCollator().collate_batch(features)


class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """
    def to(self, device):
        return self

class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """
    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)
    
    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """
    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader) 
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset) 
            for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader) 
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name]) 

class MultitaskTrainer(transformers.Trainer):

    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
              train_dataset,
              batch_size=self.args.train_batch_size,
              sampler=train_sampler
            ),
        )

        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each 
        task Dataloader
        """
        return MultitaskDataloader({
            task_name: self.get_single_train_dataloader(task_name, task_dataset)
            for task_name, task_dataset in self.train_dataset.items()
        })
    
    def train(self, *args):
      config = transformers.AutoConfig.from_pretrained("gpt2")
      model = transformers.AutoModelWithLMHead.from_pretrained("gpt2", config=config)
      self.trainer = Trainer(
        model=model,
        args=self.args,
        # transformers.TrainingArguments(
        #   output_dir="./models/multitask_model",
        #   overwrite_output_dir=True,
        #   learning_rate=1e-5,
        #   do_train=True,
        #   num_train_epochs=100,
        #   # Adjust batch size if this doesn't fit on the Colab GPU
        #   per_device_train_batch_size=8,  
        #   save_steps=3000,
        # ),
        data_collator=data_collator,
        train_dataset=train_dataset,
      )
      self.trainer.train(*args)

    # def prediction_loop(self, dataset):
    #     return self.trainer.predict(dataset)

    def compute_loss2(self, model, inputs, return_outputs=True):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        print(inputs)
        reranking_layer(outputs, inputs._get_value(), tokenizer=tokenizer) #input value is tensor
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
# %%

trainer = MultitaskTrainer(
    model=multitask_model,
    args=TRAINER_ARGS,
    data_collator=data_collator,
)

if DO_TRAIN:
    trainer.train()
# else:
#     trainer.train('./models/multitask_model_b5/checkpoint-21000/')

# %%
# From: https://discuss.huggingface.co/t/using-trainer-at-inference-time/9378/5
model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained('./pretrained_models/testtraintime')


# %%

# torch.cuda.empty_cache()
# %%
preds_dict: Dict[str, trainer_utils.PredictionOutput] = {}
for task_name in ["token", "token_type", "line"]:
    print("predicting: " + task_name)
    eval_dataloader = DataLoaderWithTaskname(
        task_name,
        trainer.get_eval_dataloader(eval_dataset=dataset_dict[task_name])
    )
    print(eval_dataloader.data_loader.collate_fn)
    preds_dict[task_name] = trainer.prediction_loop(
        eval_dataloader, 
        description=f"Validation: {task_name}",
    )
    # preds_dict[task_name] = trainer.prediction_loop(dataset_dict[task_name])


print(preds_dict)

# %%
from sklearn.metrics import accuracy_score, label_ranking_average_precision_score

accuracy_dict = {}
mrr_dict = {}

for task_name in ["token", "token_type", "line"]:
  accuracy_dict[task_name] = accuracy_score(preds_dict[task_name].predictions.flatten(),
    preds_dict[task_name].label_ids)
  
  mrr_dict[task_name] = label_ranking_average_precision_score(preds_dict[task_name].predictions.flatten(),
    preds_dict[task_name].label_ids)
  
print("accuracy_dict:")
print(accuracy_dict)
print("mrr_dict:")
print(mrr_dict)
print("preds_dict:")
print(preds_dict)

# %%