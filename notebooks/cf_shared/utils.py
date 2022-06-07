from pathlib import Path
import datetime

def get_us(start, end):
    dt = end - start
    return dt.seconds * 1e6 + dt.microseconds

def get_elapsed_us(start):
    return get_us(start, datetime.datetime.now())

def print_elapsed_seconds(start_time, label: str = "that"):
    print(label + " took: {:0.2f}s".format(get_elapsed_us(start_time)/1e6))

def get_source_file_names_from_converted_folder(converted_path, sourcefiles_path):
    start_time = datetime.datetime.now()
    converted_paths = [str(x) for x in Path(converted_path).glob("*.txt")]
    print_elapsed_seconds(start_time, "globbing converted files from disk")
    paths = [sourcefiles_path + conv_path[conv_path.rfind('/')+1:-4] for conv_path in converted_paths]
    return paths