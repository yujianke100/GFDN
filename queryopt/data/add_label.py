import pandas as pd
import os
root_path, _ = os.path.split(os.path.abspath(__file__)) 
df = pd.read_csv(root_path + '/graph.e', header=None, sep=' ')
df[2] = 1
df.to_csv(root_path + '/train.txt', header=False, index = False, sep='\t')
max_u = max(df[0]) + 1
max_v = max(df[1]) + 1
with open(root_path + "/train.meta", 'w') as f:
    f.write("{}\n{}".format(max_u, max_v))