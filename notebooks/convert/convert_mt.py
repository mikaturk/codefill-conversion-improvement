# %%
import os
import sys
from functools import wraps
import datetime
import random

sys.path.append("..")
from cf_shared.convert import convert_paths
from cf_shared.utils import get_source_and_converted_paths, print_elapsed_seconds, timed_cf_glob

DEBUG_FILENAMES = False
THREADS = 20
MAX_PATHS = -1
# 1 Megabyte (per file)
MAX_FILE_SIZE = 1_000_000
TIMES_JSON = 'experiment.json'
PY_SOURCEFILES_LOCATION = './deduplicated_code_fill_pretrain'
CONVERTED_PATH = './experiment-dataset/converted/'

os.chdir('/mnt/mturk/cf_sample_data/')

if not os.path.exists(CONVERTED_PATH):
    os.makedirs(CONVERTED_PATH)

save_stdout = sys.stdout

# %%
print("starting conversion")

paths_input = timed_cf_glob(PY_SOURCEFILES_LOCATION, "*.py*")
# paths_input = paths_input[:MAX_PATHS]

# A bit over 1000 so we're unlikely to be under 1000 files, as this is a random process
TARGET = 1011

# This 1.6M number is the approximate amount of files resulting from the conversion of all 1.8M files, # dividing by the full 1.8M files would require the target to be set a bit higher to account for
# some files failing the conversion.
threshold = TARGET / 1_603_400
paths_input = list(filter(lambda _: random.random() < threshold, paths_input))

start_time = datetime.datetime.now()
converted_paths_opt = convert_paths(paths_input, CONVERTED_PATH, times_json=TIMES_JSON, n_threads=THREADS)
print_elapsed_seconds(start_time, "converting files")
paths, converted_paths = get_source_and_converted_paths(converted_paths_opt)
