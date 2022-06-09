from pathlib import Path
import random
import shutil
import os
import sys
sys.path.append("..")
from cf_shared.utils import get_source_paths_from_converted_folder

os.chdir('/mnt/mturk/cf_sample_data/')

PY_SOURCEFILES_LOCATION = './deduplicated_code_fill_pretrain/'
sourcefiles_path = './experiment-dataset/src'
converted_path = './experiment-dataset/converted'

paths_source = get_source_paths_from_converted_folder(converted_path, PY_SOURCEFILES_LOCATION)
paths_dest = get_source_paths_from_converted_folder(converted_path, sourcefiles_path)
# for (src, dest) in zip(paths_source, paths_dest):
#   shutil.copyfile(src, dest)

print(len(paths_source))

