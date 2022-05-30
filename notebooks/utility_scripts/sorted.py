# %%
import json
import numpy as np
  
f = open('/mnt/mturk/cf_sample_data/times_all_v2.json')
data = json.load(f)
f.close()
# %%

data.sort(key=lambda x: x[1])

def rowmapper(row):
  return [row[1]/1e6, row[0], row[2]]

mpd = list(map(rowmapper, data))

seconds = sum(map(lambda x: x[1], data))/1e6
print("seconds=" + str(seconds))

# print(json.dumps(list(filter(lambda x: x[2] == "f", mpd))[-5:]))
# print(json.dumps(mpd[-10:], indent=2))
mpd[-20:]

# %%
# json.dumps(mpd[3000:3005], indent=2)

# %%

seconds = np.array(list(map(lambda x: x[1], data)))/1e6
len(seconds)
# np.max(seconds), np.min(seconds), np.median(seconds), np.average(seconds)
# %%
