import os
import shutil
import sys
from typing import List

sys.path.append("..")
from cf_shared.convert_previous import convert_optional_original
from cf_shared.utils import cf_glob
from cf_shared.convert import ConversionResult, convert_paths, get_converted_file_path_add, get_converted_file_path_replace

def test_get_converted_file_path_add():
  # Test folder name without trailing slash
  assert get_converted_file_path_add("./converted", "./sourcefiles/example.py") == "./converted/example.py.txt"
  # Test folder name with trailing slash
  assert get_converted_file_path_add("./converted/", "./sourcefiles/example.py") == "./converted/example.py.txt"
  # Test file name with multiple dots
  assert get_converted_file_path_add("./converted", "./sourcefiles/example.test.py") == "./converted/example.test.py.txt"
  # Test non-py file extension with multiple dots
  assert get_converted_file_path_add("./converted", "./sourcefiles/example.test.py-example") == "./converted/example.test.py-example.txt"

def test_get_converted_file_path_replace():
  # Test folder name without trailing slash
  assert get_converted_file_path_replace("./converted", "./sourcefiles/example.py") == "./converted/example.txt"
  # Test folder name with trailing slash
  assert get_converted_file_path_replace("./converted/", "./sourcefiles/example.py") == "./converted/example.txt"
  # Test file name with multiple dots
  assert get_converted_file_path_replace("./converted", "./sourcefiles/example.test.py") == "./converted/example.test.txt"
  # Test non-py file extension with multiple dots
  assert get_converted_file_path_replace("./converted", "./sourcefiles/example.test.py-example") == "./converted/example.test.txt"

CONVERTED_REFERENCE_FOLDER = "test-data/converted_reference"
TEST_DATASET_SOURCE_FOLDER = "test-data/source"
CONVERTED_COMPARISON_FOLDER = "test-data/converted_comparison"

def create_reference_conversion(paths: List[str]):
  os.makedirs(CONVERTED_REFERENCE_FOLDER)
  for path in paths:
    converted_path = get_converted_file_path_add(CONVERTED_REFERENCE_FOLDER, path)
    convert_optional_original(path, converted_path)

def conversion_result_is_equal(cr1: ConversionResult, cr2: ConversionResult):
  # Skip index 2, we do not require the execution time to be the same for the results to be the same
  return cr1[0] == cr2[0] and cr1[1] == cr2[1] and cr1[3] == cr2[3] and cr1[4] == cr2[4]

def test_convert_paths():
  """Integration test, tests both the batching of conversion as well as 
  the implementation of `convert` and `converts`.
  """
  shutil.rmtree(CONVERTED_REFERENCE_FOLDER, ignore_errors=True)

  paths = cf_glob(TEST_DATASET_SOURCE_FOLDER, '*.py*')

  if not os.path.exists(CONVERTED_REFERENCE_FOLDER):
    create_reference_conversion(paths)
  
  if not os.path.exists(CONVERTED_COMPARISON_FOLDER):
      os.makedirs(CONVERTED_COMPARISON_FOLDER)


  expected_conversion_results: List[ConversionResult] = [
    (
      path,
      get_converted_file_path_add(CONVERTED_COMPARISON_FOLDER, path),
      0, # Ignored by equality checker
      "s",
      None
    ) for path in paths
  ]

  conversion_results = convert_paths(paths, CONVERTED_COMPARISON_FOLDER, n_threads=2)

  for (cr1, cr2) in zip(expected_conversion_results, conversion_results):
    assert conversion_result_is_equal(cr1, cr2)

  reference_files = cf_glob(CONVERTED_REFERENCE_FOLDER, '*.txt')
  comparison_files = cf_glob(CONVERTED_COMPARISON_FOLDER, '*.txt')

  print(reference_files)
  print(comparison_files)

  for (ref_path, comp_path) in zip(reference_files, comparison_files):
    with open(ref_path, "r") as ref_fd:
      ref_text = ref_fd.read()
    with open(comp_path, "r") as comp_fd:
      comp_text = comp_fd.read()

    assert ref_text == comp_text

