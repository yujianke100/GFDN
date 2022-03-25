import numpy as np
import pyabcore
import pandas as pd
import os

root_path, _ = os.path.split(os.path.abspath(__file__)) 
df = pd.read_csv(root_path+'/data/train.txt', names=['u', 'i', 'l'], sep='\t')
max_u = max(df['u'])+1
max_v = max(df['i'])+1
edges = np.array(df[['u','i']], dtype=np.int32)
# test = pyabcore.Pyabcore('./data/train')
test = pyabcore.Pyabcore(max_u, max_v)
test.index(edges)
test.query(2, 50)
print(np.array(test.get_left()))
print(np.array(test.get_right()))
# print(len(np.array(test.get_left())))
# print(len(np.array(test.get_right())))

test2 = pyabcore.Pyabcore('./data/train')
test2.index()
test2.query(2, 50)
print(np.array(test2.get_left()))
print(np.array(test2.get_right()))