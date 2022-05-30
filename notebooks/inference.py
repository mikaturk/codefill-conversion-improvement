# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

# %%
os.chdir('/mnt/mturk/cf_sample_data/')

tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = AutoModelForCausalLM.from_pretrained("./pretrained_models/from-roberta", ignore_mismatched_sizes=True)

# %%

generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
# generator = pipeline(task="text2text-generation", model=model, tokenizer=tokenizer)

output_text = generator(
"""def multiply_numbers(a, b):\n    return a *"""
)

print(output_text)
