from pathlib import Path
import datetime
from typing import List
import os

def get_us(start: datetime.datetime, end: datetime.datetime) -> int:
    dt = end - start
    return dt.seconds * 1_000_000 + dt.microseconds

def get_elapsed_us(start: datetime.datetime) -> int:
    return get_us(start, datetime.datetime.now())

def print_elapsed_seconds(start_time: datetime.datetime, label: str = "that") -> None:
    print(label + " took: {:0.2f}s".format(get_elapsed_us(start_time)/1e6))

def get_source_paths_from_converted_folder(converted_path: str, sourcefiles_path: str):
    paths, _ = get_source_and_converted_paths(converted_path, sourcefiles_path)
    return paths

def get_source_and_converted_paths(converted_path: str, sourcefiles_path: str):
    start_time = datetime.datetime.now()
    converted_paths = [str(x) for x in Path(converted_path).glob("*.txt")]
    print_elapsed_seconds(start_time, "globbing converted files from disk")
    paths = [sourcefiles_path + conv_path[conv_path.rfind('/')+1:-4] for conv_path in converted_paths]
    return paths, converted_paths

def cf_glob(path: str, pattern: str) -> List[str]:
    return [str(x) for x in Path(path).glob(pattern)]

def timed_cf_glob(path: str, pattern: str) -> List[str]:
    start_time = datetime.datetime.now()
    paths = cf_glob(path, pattern)
    print_elapsed_seconds(start_time, "globbing {} files from disk".format(len(paths)))
    return paths

def get_filesizes(paths: List[str]) -> List[int]:
    return list(map(os.path.getsize, paths))

def filter_max_filesize(paths: List[str], filesizes: List[int], max_filesize: int) -> List[str]:
    return [path for (path, filesize) in zip(paths, filesizes) if filesize < max_filesize]

