# %% [markdown]
# Install the correct dependencies on HuggingFace transformer and ternsorflow

# %%
# We won't need TensorFlow here
# %pip uninstall -y tensorflow
# Install `transformers` from master
# %pip install git+https://github.com/huggingface/transformers
# %pip list | grep -E 'transformers|tokenizers'
# %pip install nlp==0.2.0
# %pip install datasets
# %pip install git+https://github.com/huggingface/nlp

# transformers version at notebook update --- 2.11.0
# tokenizers version at notebook update --- 0.8.0rc1

# %% [markdown]
# Fetch datasets

# %%
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

def multireplace(string, replacements, ignore_case=False):
    """
    Given a string and a replacement map, it returns the replaced string.
    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :param bool ignore_case: whether the match should be case insensitive
    :rtype: str
    """
    # If case insensitive, we need to normalize the old string so that later a replacement
    # can be found. For instance with {"HEY": "lol"} we should match and find a replacement for "hey",
    # "HEY", "hEy", etc.
    if ignore_case:
        def normalize_old(s):
            return s.lower()
        re_mode = re.IGNORECASE
    else:
        def normalize_old(s):
            return s
        re_mode = 0

    replacements = {normalize_old(key): val for key, val in replacements.items()}
    
    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
    # 'hey ABC' and not 'hey ABc'
    rep_sorted = sorted(replacements, key=len, reverse=True)
    rep_escaped = map(re.escape, rep_sorted)
    
    # Create a big OR regex that matches any of the substrings to replace
    pattern = re.compile("|".join(rep_escaped), re_mode)
    
    # For each match, look up the new string in the replacements, being the key the normalized old string
    return pattern.sub(lambda match: replacements[normalize_old(match.group(0))], string)


def convert(file, output_file):
    with open (file, "r") as f:
        text = f.read()  

    replacements = {}
    for node in ast.iter_child_nodes(ast.parse(text)):
        if isinstance(node, ast.ImportFrom):
            replacements.update({node.module: 'MODULE'})
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            for i, v in enumerate(node.names):
                if(node.names[i].asname):
                    replacements.update({node.names[i].name: 'LIB'})                
                    replacements.update({node.names[i].asname: 'ALIAS'})
                else:
                    replacements.update({node.names[i].name: 'LIBRARY'})


    # reomve * from the dictionary (handle from module import * statement)
    replacements.pop('*', None)
    print('List of modules and libraries to replace:\n', replacements)

    with open('med.py','w') as f:
        f.write(multireplace(text, replacements, ignore_case = True))

    file = 'med.py'
    with open(file,'rb') as f:
        tokens = list(tokenize.tokenize(f.readline))
        
    ### extract important data from the output of tokenize package
    # toks = pd.DataFrame(columns = ['original','type','text', 'line','pos'])

    last_line = 0
    last_pos = 0
    tokss = [] # PERF
    toks_index = 0 # PERF
    for token in tokens:
        
        tok_org = token.string
        tok_text = token.string    
        tok_type = str(token).split('(')[2].split(')')[0]

        # convert keywords to upper
        if keyword.iskeyword(tok_text):
            tok_type = str.upper(tok_text)
        
        #extract operations
        # if tok_type == 'OP':
        #     tok_type = tok_text


        # getting rid of comments and empty lines
        if tok_type in ['NL','NEWLINE','COMMENT']:
            continue
        
        #retrieve the position
        tok_line = token.start[0]
        
        if last_line == tok_line:
            last_pos +=  1
        else:
            last_pos = 1
        tok_pos = last_pos
        last_line = tok_line
        
        # toks = toks.append({'type':tok_type,
        #                     'original':tok_org,
        #                     'text':tok_text,
        #                     'line':tok_line,
        #                     'pos':tok_pos},ignore_index=True)
        # tokss.append(pd.DataFrame({'type':tok_type,
        #                     'original':tok_org,
        #                     'text':tok_text,
        #                     'line':tok_line,
        #                     'pos':tok_pos}, index=[toks_index]))
        tokss.append({'type':tok_type,
                            'original':tok_org,
                            'text':tok_text,
                            'line':tok_line,
                            'pos':tok_pos})
        # toks_index += 1 # PERF

    # toks = pd.concat(tokss)
    toks = pd.DataFrame(tokss)

    # remove encoding lines and end of file
    toks.line = toks.line.astype('int')
    toks.pos = toks.pos.astype('int')
    toks = toks.loc[~((toks.type == 'ENCODING') | (toks.type == 'ENDMARKER'))]
    toks['doc'] = (toks.text.str.contains('"""') | toks.text.str.contains("'''"))
    toks = toks.loc[~(toks.doc)].drop(['doc'],axis=1)

    toks.head(20)

    indent = 0
    last_line = 0

    tokss = [] # PERF

    # for index,row in toks.iterrows():
    for row in toks.itertuples():
        if row.type == "INDENT":
            indent +=1
            continue
        if row.type == "DEDENT":
            indent -=1
            continue
        if row.line != last_line:
            last_line = row.line
            # toks = toks.append({'type':'\n'+indent*'\t',
            #                     'text':'\n'+indent*'\t',
            #                     'line':row.line,
            #                     'pos':row.pos-1},ignore_index=True)
            # tokss.append(pd.DataFrame({'type':'\n'+indent*'\t',
            #                     'text':'\n'+indent*'\t',
            #                     'line':row.line,
            #                     'pos':row.pos-1}, index=[toks_index]))
            tokss.append({'type':'\n'+indent*'\t',
                                'text':'\n'+indent*'\t',
                                'line':row.line,
                                'pos':row.pos-1})
    
    toks = pd.concat([toks, pd.DataFrame(tokss)])

    toks = toks.loc[~((toks.type=='INDENT') | (toks.type=='DEDENT'))]
    toks = toks.sort_values(['line','pos']).reset_index(drop=True)


    # drop the first row (empty line)
    toks.drop(toks.index[:1], inplace=True)

    toks.head(20)

    with open(file,'r') as f:
        src = f.read()

    stdout_backup = sys.stdout
    sys.stdout = open('dis.txt','w')
    dis.dis(src)
    sys.stdout = stdout_backup

    with open('dis.txt','r') as f:
        lines = f.readlines()

    # find global variables
    glbls = [].copy()    
    for l in lines:
        clean = l.replace('>>',' ').strip().split()
        if len(clean):
            try:
                int(clean[1])
                line = int(clean[0])
            except:
                clean = [str(line)]+clean
            if 'LOAD_GLOBAL' in clean:
                print('found a global!')
                glbls.append((int(clean[0]),clean[-1].replace('(','').replace(')','')))

    for l,n in glbls:
        toks.loc[(toks.line==l) & (toks.text==n),'type'] = 'GLOBAL_VARIABLE'

    toks .head(10) 

    text_imports = ' '.join(list(toks.text)).replace('\n ','\n').replace(' \n','\n').replace('\t ','\t').replace(' . ','.').replace(' (','(')
    text_imports = multireplace(text_imports, replacements, ignore_case = True)

    with open('normalized_textual_file.py','w') as f:
        f.write(text_imports)

    toks.type = toks.apply(lambda x: x['text'] if str(x['text']) in ['LIBRARY','LIB','ALIAS','MODULE'] else x['type'], axis = 1)
    code_converted = ' '.join(list(toks.type)).replace('\n ','\n').replace(' \n','\n').replace('\t ','\t').replace(' . ','.').replace(' (','(')

    final_replacements = {'GLOBAL_VARIABLE(':'FUNCTION_CALL(',                      
    #                       'NAME.NAME':'NAME',
                          'NAME(':'FUNCTION_CALL(',
                          'NAME':'LOCAL_VARIABLE'}

    code_converted = multireplace(code_converted, final_replacements, ignore_case = False)

    with open(output_file,'w') as f:
        f.write(code_converted)


WEIGHT_MATRIX = {
        'NUMBER' : [1.625, 1.25, 1.125],
        'NAME' : [1.625, 1.125, 1.5],
        'LOCAL_VARIABLE' : [1.625, 1.125, 1.5],
        'FUNCTION_NAME' : [1.625, 1.25, 1.5]
    }

input_file = "/tmp/input_file.txt"
output_file = "/tmp/output_file.txt"
def reranking_layer(outputs, context, tokenizer):

  with open(input_file, 'w') as f:
    f.write(context);
  
  convert(file_path=input_file, output_file=output_file)
  with open(output_file, 'rb') as context:
    inputs = list(zip(tokenizer(input_file), tokenizer(output_file)))
    for item in inputs:
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(WEIGHT_MATRIX[item[1]]))


# %%
# convert("./sample_data/data/peakfinder.py", "./converted_train.txt")

# %%
%load_ext line_profiler
# %lprun -f convert convert("sample_data/data/raw_to_mat.py", "sample_data/converted/raw_to_mat.txt")
%lprun -f convert convert("sample_data/data/0002_add_new_column_conference.py", "sample_data/converted/0002_add_new_column_conference.txt")

# convert("sample_data/data/raw_to_mat.py", "sample_data/converted/raw_to_mat.txt")

# %%
from pathlib import Path
import glob
def convert_n(n_paths):
    paths = [str(x) for x in Path(".").glob("./sample_data/data/*.py")]
    paths = paths[:n_paths]
    converted_paths = []
    for path in paths:
        converted_path = "./sample_data/converted/"+ path.split("/").pop().split(".")[0] + ".txt"
        try:
            convert(path, converted_path)
            converted_paths.append(converted_path)
        except:
            pass

# %load_ext line_profiler
# %lprun -f convert convert("sample_data/data/porndl.py", "sample_data/converted/porndl.txt")
# %lprun -f convert convert("sample_data/data/raw_to_mat.py", "sample_data/converted/raw_to_mat.txt")
# %lprun -f convert convert_n(200)

# %%
# pretrain dataset
# !wget https://huggingface.co/rgismondi/python-50k-dedup/blob/main/pretrain_dataset.zip
# !wget https://huggingface.co/rgismondi/python-50k-dedup/resolve/main/pretrain_dataset.zip
# !unzip 'pretrain_dataset.zip'

# converted dataset
# ! wget https://huggingface.co/rgismondi/python-50k-dedup/blob/main/converted_dataset.zip
# ! wget https://huggingface.co/rgismondi/python-50k-dedup/resolve/main/converted_dataset.zip
# ! unzip 'converted_dataset.zip'

# test dataset
# !wget https://huggingface.co/rgismondi/python-50k-dedup/blob/main/finetune_eval_dataset.zip
# !wget https://huggingface.co/rgismondi/python-50k-dedup/resolve/main/finetune_eval_dataset.zip
# !unzip 'finetune_eval_dataset.zip'

# %% [markdown]
# Train a customised python byte-level Byte-pair encoding tokenizer. 

# %%
from pathlib import Path
from transformers import AutoTokenizer,TextDataset,DataCollatorForLanguageModeling
import glob
import random 

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# %%
paths = [str(x) for x in Path(".").glob("./sample_data/data/*.py")]
n_paths = 200
paths = paths[:n_paths]
converted_paths = []
i = 0
for path in paths:
  converted_path = "./sample_data/converted/"+ path.split("/").pop().split(".")[0] + ".txt"
  # print(converted_path)
  try:
    convert(path, converted_path)
    converted_paths.append(converted_path)
  except:
    pass
# converted_paths = ["./sample_data/converted/"+ path.split("/").pop().split(".")[0] + ".txt" for path in paths]
# paths
# for (path, conv_path) in zip(paths, converted_paths):
#     try:
#       convert(path, conv_path)
#     except:
#       pass

# convert is too sequential, this will not work
# Parallel(n_jobs=-1)(delayed(convert)(path, conv_path) for (path, conv_path) in zip(paths, converted_paths))

with open("./train.txt", "wb") as train_outfile:
  with open("./test.txt", "wb") as test_outfile:
    for f in paths:
        choice = random.random()
        with open(f, "rb") as infile:
            if choice > 0.1:
              train_outfile.write(infile.read())
            else:
              test_outfile.write(infile.read())

with open("./converted_train.txt", "wb") as train_outfile:
  with open("./converted_test.txt", "wb") as test_outfile:
    for f in converted_paths:
        choice = random.random()
        with open(f, "rb") as infile:
            if choice > 0.1:
              train_outfile.write(infile.read())
            else:
              test_outfile.write(infile.read())


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
#pretrain_raw_files = glob.glob("./pretrain_dataset" + '/**/*.py', recursive=True)
#pretrain_converted_files = glob.glob("./pretrain_converted_dataset" + '/**/*.py', recursive=True)
print("converted!")
# %%
# tokenizer("for i in range(10)")["input_ids"]

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
    "token_type": train_dataset,
    "line": train_dataset,
}

print(dataset_dict["token"])

print("loaded configs!")

# %%
from transformers.utils.dummy_pt_objects import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from transformers import GPT2Config, EncoderDecoderConfig, EncoderDecoderModel


class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    def _get_models(self):
      return self.taskmodels_dict

    @classmethod
    def create(cls, model_name, model_type_dict, model_config_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models. 

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained( "gpt2",
                config=model_config_dict[task_name],
            )
            if shared_encoder is None:
                shared_encoder = cls.get_encoder(model)
            else:
                setattr(model, "encoder", shared_encoder)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)
    

    @classmethod
    def get_encoder(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Roberta"):
            return "roberta-base"
        elif model_class_name.startswith("GPT2"):
            config = EncoderDecoderConfig.from_encoder_decoder_configs(model.config, model.config) 
            encoder_decoder = EncoderDecoderModel(config=config)
            return encoder_decoder.config.encoder
        else:
            raise KeyError(f"Add support for new model {model_class_name}")
    
    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)

# %%
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
    
    def train(self, model_name):
    # def train(self):
      config = transformers.AutoConfig.from_pretrained("gpt2")
      model = transformers.AutoModelWithLMHead.from_pretrained("gpt2", config=config)
      trainer = Trainer(
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
      trainer.train(model_name)
    #   trainer.train()

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


# %%
trainer = MultitaskTrainer(
    model=multitask_model,
    args=transformers.TrainingArguments(
        output_dir="./models/multitask_model",
        # overwrite_output_dir=True,
        overwrite_output_dir=False,
        learning_rate=1e-5,
        # do_train=True,
        do_train=False,
        num_train_epochs=100,
        # Adjust batch size if this doesn't fit on the Colab GPU
        per_device_train_batch_size=6,  
        save_steps=3000,
        eval_accumulation_steps=8
    ),
    data_collator=data_collator,
)
# trainer.train()
trainer.train('./models_6g_acc/multitask_model/checkpoint-12000')

# %%

import gc

gc.collect()

torch.cuda.empty_cache()
# %%
preds_dict = {}
for task_name in ["token", "token_type", "line"]:
    eval_dataloader = DataLoaderWithTaskname(
        task_name,
        trainer.get_eval_dataloader(eval_dataset=dataset_dict[task_name])
    )
    print(eval_dataloader.data_loader.collate_fn)
    preds_dict[task_name] = trainer.prediction_loop(
        eval_dataloader, 
        description=f"Validation: {task_name}",
    )


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
  



# %%
