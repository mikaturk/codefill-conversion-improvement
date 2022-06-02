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
# %pip install sklearn

# transformers version at notebook update --- 2.11.0
# tokenizers version at notebook update --- 0.8.0rc1

# %% [markdown]
# Fetch datasets

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
from transformers import AutoTokenizer, RobertaForQuestionAnswering,TextDataset,DataCollatorForLanguageModeling, trainer_utils
import glob
import random
from cf_shared.utils import get_source_file_names_from_converted_folder, get_elapsed_us, print_elapsed_seconds
from cf_shared.MultitaskModel import MultitaskModel

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

DEBUG_FILENAMES = False
PERFORM_CONVERSION = False
PERFORM_DATASET_COPY = False
THREADS = 20
MAX_PATHS = 10_000
TIMES_JSON = 'times_script_10k.json'
PY_SOURCEFILES_LOCATION = './deduplicated_code_fill_pretrain/'

os.chdir('/mnt/mturk/cf_sample_data/')

converted_path = './converted_all_lt_1mb/'
if not os.path.exists(converted_path):
    os.makedirs(converted_path)

save_stdout = sys.stdout

def multireplace(string, replacements, ignore_case=False):
    """
    Given a string and a replacement map, it returns the replaced string.
    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :param bool ignore_case: whether the match should be case insensitive
    :rtype: str
    """
    if replacements == {}:
        return string
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
    if DEBUG_FILENAMES: print("starting "+output_file, file=save_stdout)
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
    # print('List of modules and libraries to replace:\n', replacements)

    med = multireplace(text, replacements, ignore_case = True)

    tokens = tokenize.tokenize(BytesIO(med.encode('utf-8')).readline)
        
    ### extract important data from the output of tokenize package

    last_line = 0
    last_pos = 0
    tokss = []
    for token in tokens:
        
        tok_org = token.string
        tok_text = token.string    
        # tok_type = str(token).split('(')[2].split(')')[0]
        tok_type = tokenize.tok_name[token.type]

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
        
        tokss.append({'type':tok_type,
                            'original':tok_org,
                            'text':tok_text,
                            'line':tok_line,
                            'pos':tok_pos})

    toks = pd.DataFrame(tokss)

    # remove encoding lines and end of file
    toks.line = toks.line.astype('int')
    toks.pos = toks.pos.astype('int')
    toks = toks.loc[~((toks.type == 'ENCODING') | (toks.type == 'ENDMARKER'))]
    toks['doc'] = (toks.text.str.contains('"""') | toks.text.str.contains("'''"))
    toks = toks.loc[~(toks.doc)].drop(['doc'],axis=1)

    indent = 0
    last_line = 0

    tokss = [] # PERF

    for row in toks.itertuples():
        if row.type == "INDENT":
            indent +=1
            continue
        if row.type == "DEDENT":
            indent -=1
            continue
        if row.line != last_line:
            last_line = row.line
            tokss.append({'type':'\n'+indent*'\t',
                                'text':'\n'+indent*'\t',
                                'line':row.line,
                                'pos':row.pos-1})
    
    toks = pd.concat([toks, pd.DataFrame(tokss)])

    toks = toks.loc[~((toks.type=='INDENT') | (toks.type=='DEDENT'))]
    toks = toks.sort_values(['line','pos']).reset_index(drop=True)

    # drop the first row (empty line)
    toks.drop(toks.index[:1], inplace=True)

    src = text
    stdout_backup = sys.stdout
    dis_result = StringIO()
    sys.stdout = dis_result
    dis.dis(src)
    sys.stdout = stdout_backup

    lines = dis_result.getvalue().split('\n')

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
                # print('found a global!')
                glbls.append((int(clean[0]),clean[-1].replace('(','').replace(')','')))

    for l,n in glbls:
        line_eq = toks.loc[toks.line==l]
        line_eq.loc[line_eq.text==n, 'type'] = 'GLOBAL_VARIABLE'
        
    # text_imports = ' '.join(list(toks.text)).replace('\n ','\n').replace(' \n','\n').replace('\t ','\t').replace(' . ','.').replace(' (','(')
    # text_imports = multireplace(text_imports, replacements, ignore_case = True)

    # with open('normalized_textual_file.py','w') as f:
    #     f.write(text_imports)

    
    toks.loc[toks['text'].isin(['LIBRARY','LIB','ALIAS','MODULE']), 'type'] = toks['text']

    code_converted = ' '.join(list(toks.type)).replace('\n ','\n').replace(' \n','\n').replace('\t ','\t').replace(' . ','.').replace(' (','(')

    final_replacements = {'GLOBAL_VARIABLE(':'FUNCTION_CALL(',                      
    #                       'NAME.NAME':'NAME',
                          'NAME(':'FUNCTION_CALL(',
                          'NAME':'LOCAL_VARIABLE'}

    code_converted = multireplace(code_converted, final_replacements, ignore_case = False)

    with open(output_file,'w') as f:
        f.write(code_converted)
    if DEBUG_FILENAMES: print("finished "+output_file, file=save_stdout)

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
    f.write(context)
  
  convert(file_path=input_file, output_file=output_file)
  with open(output_file, 'rb') as context:
    inputs = list(zip(tokenizer(input_file), tokenizer(output_file)))
    for item in inputs:
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(WEIGHT_MATRIX[item[1]]))

# %%
# %load_ext line_profiler
# %lprun -f convert convert("sample_data/data/raw_to_mat.py", "sample_data/converted/raw_to_mat.txt")
# %lprun -f convert convert("sample_data/data/0002_add_new_column_conference.py", "sample_data/converted/0002_add_new_column_conference.txt")


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

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# %%

def convert_optional(path, converted_path):
    # Uncomment when using convert_new    
    # tmp_dir = tempfile.mkdtemp()

    try:
        b4 = datetime.datetime.now()

        # convert_new(path, converted_path, tmp_dir)
        convert(path, converted_path)

        # Uncomment when using convert_new    
        # shutil.rmtree(tmp_dir)
        return (path, converted_path, get_elapsed_us(b4), "s")
    except:
        # Uncomment when using convert_new    
        # shutil.rmtree(tmp_dir)
        return (path, converted_path, get_elapsed_us(b4), "f")

def convert_paths(paths):
    converted_paths_before = []
    for path in paths:
        file_name = path.split("/").pop()
        # Enable this option for:
        # hello.py -> hello.txt
        # base_name = file_name[:file_name.rfind('.')]

        # Enable this option for:
        # hello.py -> hello.py.txt
        base_name = file_name
        converted_paths_before.append(converted_path + base_name + ".txt")
    print("CONVERTING {} PYTHON FILES".format(len(paths)))
    converted_paths_opt = Parallel(n_jobs=THREADS)(delayed(convert_optional)(path, conv_path) for (path, conv_path) in zip(paths, converted_paths_before))
    with open(TIMES_JSON,'w') as fd:
        fd.write(json.dumps(converted_paths_opt))
        # fd.write('[\n'+',\n'.join(map(lambda x: "[\"{}\",{}]".format(*x),converted_paths_opt))+'\n]')
    converted_paths_filtered = list(filter(lambda x: x[-1] == "s", converted_paths_opt))
    sys.stdout = save_stdout
    print("RESULT: {} FILES IN, {} FILES OUT".format(len(converted_paths_before), len(converted_paths_filtered)))
    return converted_paths_filtered

if PERFORM_CONVERSION:
    print("starting conversion")
    start_time = datetime.datetime.now()
    paths_input = [str(x) for x in Path(PY_SOURCEFILES_LOCATION).glob("*.py*")]
    paths_input = paths_input[:MAX_PATHS]
    print("globbing files from disk took: {:0.2f}s".format(get_elapsed_us(start_time)/1e6))
    start_time = datetime.datetime.now()
    converted_paths_filtered = convert_paths(paths_input)
    print("converting files took: {:0.2f}s".format(get_elapsed_us(start_time)/1e6))
    paths = list(map(lambda x: x[0], converted_paths_filtered))
    converted_paths = list(map(lambda x: x[1], converted_paths_filtered))
else:
    print("skipping conversion")
    get_source_file_names_from_converted_folder(converted_path, PY_SOURCEFILES_LOCATION)


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
    print("the writing of source files took: {:0.2f}s".format(get_elapsed_us(start_time)/1e6))
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
# TODO: Ask what these lines are for.
# pretrain_raw_files = glob.glob("./pretrain_dataset" + '/**/*.py', recursive=True)
# pretrain_converted_files = glob.glob("./pretrain_converted_dataset" + '/**/*.py', recursive=True)

# raise Exception("stopping the script...")
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
DO_TRAIN = True

trainer = MultitaskTrainer(
    model=multitask_model,
    args=transformers.TrainingArguments(
        output_dir="./models/multitask_model_testtraintime",
        overwrite_output_dir=DO_TRAIN,
        learning_rate=1e-5,
        do_train=DO_TRAIN,
        num_train_epochs=100,
        # Adjust batch size if this doesn't fit on the Colab GPU
        per_device_train_batch_size=24,  
        save_steps=110000,
        fp16=True,
        eval_accumulation_steps=8
    ),
    data_collator=data_collator,
)

if DO_TRAIN:
    trainer.train()
else:
    trainer.train('./models/multitask_model_b5/checkpoint-21000/')

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

# %%
list