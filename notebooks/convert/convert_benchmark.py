
# %%

import os
import sys
from functools import wraps

sys.path.append("..")
# %%
import cf_shared.convert_previous as convert_prev
import cf_shared.convert as convert_cur
from cf_shared.utils import cf_glob

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

os.chdir('/home/mturk/rp/codefill/notebooks/cf_shared/test-data')

# Use a python debugger to investigate the global variable bug

# convert_prev.convert_original("source/global.py", "converted_reference/global.py.txt")
convert_cur.convert("source/global.py", "converted_reference/global.py.txt")
# %%
# os.chdir('/mnt/mturk/cf_sample_data/')
os.chdir('/home/mturk/rp/codefill/notebooks')



# %%
# %load_ext line_profiler

# %lprun -f convert_prev.convert_old convert_prev.convert_old("slowfiles/test_randomstate.py", "slowfiles/test_randomstate.txt")
# %lprun -f convert_cur.converts convert_cur.convert("slowfiles/test_randomstate.py", "slowfiles/test_randomstate.txt")
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
