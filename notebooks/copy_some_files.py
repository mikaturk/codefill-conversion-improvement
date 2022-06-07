from pathlib import Path
import random
import shutil
import os
from cf_shared.utils import get_source_file_names_from_converted_folder

os.chdir('/mnt/mturk/cf_sample_data/')

PY_SOURCEFILES_LOCATION = './deduplicated_code_fill_pretrain/'
converted_path = './converted_mali-100-pyfiles'
sourcefiles_path = './mali-100-pyfiles/'

paths_source = get_source_file_names_from_converted_folder(converted_path, PY_SOURCEFILES_LOCATION)
paths_dest = get_source_file_names_from_converted_folder(converted_path, sourcefiles_path)
for (src, dest) in zip(paths_source, paths_dest):
  shutil.copyfile(src, dest)
