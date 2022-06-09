import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

os.chdir('/mnt/mturk/cf_sample_data/')

sourcefiles_path = './experiment-dataset/src'
converted_path = './experiment-dataset/converted'

X = list(Path(sourcefiles_path).glob('*.py*'))
y = list(Path(converted_path).glob('*.txt'))

X_train_path = './experiment-dataset/src-train'
X_test_path = './experiment-dataset/src-test'
y_train_path = './experiment-dataset/converted-train'
y_test_path = './experiment-dataset/converted-test'

print(len(X))
print(len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

for p in X_train:
  shutil.copy(p, X_train_path)

for p in X_test:
  shutil.copy(p, X_test_path)

for p in y_train:
  shutil.copy(p, y_train_path)

for p in y_test:
  shutil.copy(p, y_test_path)

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))
