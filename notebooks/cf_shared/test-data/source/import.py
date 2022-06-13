from os import path
import json

print(path.join("./my_folder", "file.csv"))

obj = json.loads('{"hi": 3}')

print(obj["hi"])