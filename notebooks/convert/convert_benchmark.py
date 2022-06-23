
# %%

import datetime
import json
import os
import sys
from functools import wraps
from tempfile import mkdtemp

# Add the parent folder to the path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# %%
import cf_shared.convert_previous as convert_prev
import cf_shared.convert as convert_cur
from cf_shared.utils import cf_glob, print_elapsed_seconds, get_elapsed_us

os.chdir('/mnt/mturk/cf_sample_data/')

debug_filenames = True
save_stdout = sys.stdout

# %% 
def convert_n(n_paths, conv):
    paths = cf_glob(".", "./sample_data/data/*.py")
    paths = paths[:n_paths]
    converted_paths = []
    for path in paths:
        converted_path = "./sample_data/converted/"+ path.split("/").pop().split(".")[0] + ".txt"
        try:
            conv(path, converted_path)
            converted_paths.append(converted_path)
        except:
            pass


# %%

def convert_n_times(n, path, converted_path, conv_optional, times_json=None):
    """
    Runs the convert function multiple times to get more consistent results
    """
    times = []
    for i in range(n):
        print("starting run #{0}".format(i))
        result = conv_optional(path, converted_path)
        print(result)
        times.append(result[2])

    return times

os.chdir('/home/mturk/rp/codefill/notebooks')

# %%

times_original = convert_n_times(5, "slowfiles/test_randomstate.py", "slowfiles/test_randomstate.txt", convert_prev.convert_optional_old)
times_old = convert_n_times(5, "slowfiles/test_randomstate.py", "slowfiles/test_randomstate.txt", convert_prev.convert_optional_old)
times_new = convert_n_times(5, "slowfiles/test_randomstate.py", "slowfiles/test_randomstate.txt", convert_cur.convert_optional)
print({'original':times_original, 'old':times_old, 'new':times_new})
# %%

# os.chdir('/home/mturk/rp/codefill/notebooks/cf_shared/test-data')

# Use a python debugger to investigate the global variable bug
# Update: bug has since been fixed but you can still use this example to track down any other bugs.

# convert_prev.convert_original("source/global.py", "converted_reference/global.py.txt")
# convert_cur.convert("source/global.py", "converted_reference/global.py.txt")
# %%
# os.chdir('/mnt/mturk/cf_sample_data/')


"""
The next two cells contain code that only works in jupyter notebooks, this file can be executed as a jupyter notebook in visual studio code.
"""
# %%
%load_ext line_profiler

# %%
os.chdir('/home/mturk/rp/codefill/notebooks/')

# %lprun -f convert_prev.convert_original convert_prev.convert_original("slowfiles/test_randomstate.py", "slowfiles/test_randomstate.txt")
%lprun -f convert_prev.convert_old convert_prev.convert_optional_old("slowfiles/test_randomstate.py", "slowfiles/test_randomstate.txt")
# %lprun -f convert_cur.converts convert_cur.convert_optional("slowfiles/pydoc_topics.py-x068.py-Gadgetoid", "slowfiles/pydoc_topics.py-x068.py-Gadgetoid.txt")
# %lprun -f convert_cur.converts convert_cur.convert_optional("slowfiles/test_randomstate.py", "slowfiles/test_randomstate.txt")
# %lprun -f convert_prev.convert_old convert_prev.convert_old("slowfiles/drive_v2.py", "slowfiles/drive_v2.txt")
# %lprun -f convert_cur.converts convert_cur.convert("slowfiles/drive_v2.py", "slowfiles/drive_v2.txt")
# %lprun -f convert_cur.converts convert_cur.convert("slowfiles/rpdb2.py", "slowfiles/rpdb2.txt")
# %lprun -f convert_cur.converts convert_cur.convert("slowfiles/returns_vec4.py", "slowfiles/returns_vec4.txt")
# %lprun -f convert_cur.converts convert_cur.convert("slowfiles/sp_applications.py", "slowfiles/sp_applications.txt")
# %lprun -f convert_cur.converts convert_cur.convert("slowfiles/data_se.py", "slowfiles/data_se.txt")
# %lprun -f convert_cur.converts convert_cur.convert("slowfiles/145.py", "slowfiles/145.txt")
# %lprun -f convert convert_n(1, convert_prev.convert_original)

# %%
# %lprun -f convert_cur.converts convert_cur.convert("slowfiles/145.py", "slowfiles/145.txt")

# %%

# convert_cur.convert("slowfiles/145.py", "slowfiles/145.txt")
# convert_cur.convert("slowfiles/BC_WA_policy.py", "slowfiles/BC_WA_policy.txt")
# convert_cur.convert("./test_example.txt", "./converted_test_example.txt")

# %%

# convert_prev.convert_original("source/global.py", "converted_reference/global.py.txt")

# convert_cur.convert("source/global.py", "converted_comparison/global.py.txt")
# %%
os.chdir('/home/mturk/rp/codefill/notebooks')

pyfile = 'test_randomstate.py'

# start_time = datetime.datetime.now()
# res = convert_prev.convert_optional_original("slowfiles/"+pyfile, "slowfiles/"+pyfile+".txt")
# print_elapsed_seconds(start_time, "og conversion of : " + pyfile)

count_old = 0
total_time_old = 0
count_new = 0
total_time_new = 0

# %%
start_time = datetime.datetime.now()
count_old += 1
res = convert_prev.convert_optional_original("slowfiles/"+pyfile, "slowfiles/"+pyfile+".txt")
total_time_old += get_elapsed_us(start_time)
print("OLD: {:.2f} seconds averaged over {} runs".format(total_time_old / count_old / 1e6, count_old))

# %%
res
# %%

start_time = datetime.datetime.now()
count_new += 1
res = convert_cur.convert_optional("slowfiles/"+pyfile, "slowfiles/"+pyfile+".txt")
total_time_new += get_elapsed_us(start_time)
print("NEW: {:.2f} seconds averaged over {} runs".format(total_time_new / count_new / 1e6, count_new))


# %%
