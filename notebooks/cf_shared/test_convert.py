import sys
sys.path.append("..")
from cf_shared.convert import get_converted_file_path_add

def test_get_converted_file_path_add():
  assert get_converted_file_path_add("./converted", "./sourcefiles/example.py") == "./converted/example.py.txt"
  assert get_converted_file_path_add("./converted/", "./sourcefiles/example.py") == "./converted/example.py.txt"
