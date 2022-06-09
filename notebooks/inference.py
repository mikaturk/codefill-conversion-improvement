# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GPT2LMHeadModel, GPT2Config, AutoModelForSequenceClassification, DataCollatorForLanguageModeling, Trainer, TrainingArguments, PreTrainedTokenizerFast
import os
from pathlib import Path
import torch
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from torch.utils.data import Dataset
import random
import datetime
from cf_shared.utils import get_source_file_names_from_converted_folder, print_elapsed_seconds
from cf_shared.MultitaskModel import MultitaskModel


os.chdir('/mnt/mturk/cf_sample_data/')

# %%

converted_path = "./converted_all_lt_1mb/"
sourcefiles_path = "./deduplicated_code_fill_pretrain/"
# model_location = "./pretrained_models/from-roberta"
model_location = "./pretrained_models/one-point-six-million"
tokenizer_model_location = "./tokenizer-model/"
tokenizer_model_location_json = tokenizer_model_location + "tokenizer.json"


DO_TRAIN_TOKENIZER = False

APPROXIMATE_FILE_AMOUNT = 20_000


if DO_TRAIN_TOKENIZER:
    paths_from_disk = get_source_file_names_from_converted_folder(converted_path, sourcefiles_path)
    paths = paths_from_disk

    threshold = APPROXIMATE_FILE_AMOUNT / 1_603_400

    paths = list(filter(lambda _: random.random() < threshold, paths))


    print('first 5 paths:')
    print(paths[:5])
    print('total paths: ' + str(len(paths)))
    tokenizer = Tokenizer(BPE())
    start_time = datetime.datetime.now()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train(files=["training_set"], trainer=trainer)
    # tokenizer.train(files=paths, vocab_size=len(paths), min_frequency=2, special_tokens=[
    #     "<s>",
    #     "<pad>",
    #     "</s>",
    #     "<unk>",
    #     "<mask>",
    # ])

    print_elapsed_seconds(start_time, "training")

    if not os.path.exists(tokenizer_model_location):
        os.makedirs(tokenizer_model_location)
    # tokenizer.save_model(tokenizer_model_location, "codecompletion")
    tokenizer.save(tokenizer_model_location_json)
else:
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_location)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_model_location_json)

# %%

# model = AutoModelForSequenceClassification.from_pretrained(model_location, config=GPT2Config())
model = GPT2LMHeadModel.from_pretrained(model_location, config=GPT2Config())
# task_name = "line"
# model = multitaskmodel.taskmodels_dict[task_name]

model.summary()
# %%
raise Exception("stopping the script early...")
generator = pipeline(task="token", model=model, tokenizer=tokenizer)
# generator = pipeline(task="text2text-generation", model=model, tokenizer=tokenizer)

output_text = generator(
"""def multiply_numbers(a, b):\n    return """
)

print(output_text)
# %%
