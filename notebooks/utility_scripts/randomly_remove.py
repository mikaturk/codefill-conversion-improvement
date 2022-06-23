from pathlib import Path
import random
import shutil
import os
from random import randrange

# Script removes converted files from a folder randomly until the target is reached.
os.chdir('/mnt/mturk/cf_sample_data/')

converted_path = './experiment-dataset/converted'

TARGET = 1000
paths = list(Path(converted_path).glob('*.txt'))
while len(paths) > TARGET:
  l = len(paths)
  ind = randrange(l)
  os.remove(paths[ind])
  del paths[ind]

