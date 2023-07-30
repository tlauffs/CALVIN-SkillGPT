import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

datapath_hulk = 'test_data/D_D/task_D_D_episode.npz'
datapath_hulk2 = 'test_data/Hulc2/hulc_2_episode.npz'
datapath_hulk2_an = 'test_data/Hulc2/single/auto_lang_ann.npy'
datapath_hulk2_an2 = 'test_data/Hulc2/single/auto_lang_ann.pckl'


d_hulk = np.load(datapath_hulk)
print(list(d_hulk.keys()))

for k in d_hulk.keys():
     print(k, d_hulk[k].shape)


d_hulk2 = np.load(datapath_hulk2)
print('\n\n\n',list(d_hulk2.keys()))

for k in d_hulk2.keys():
     print(k, d_hulk2[k].shape)


d_hulk_ann = np.load(datapath_hulk2_an, allow_pickle=True)
print(d_hulk_ann.dtype)
print(d_hulk_ann)
