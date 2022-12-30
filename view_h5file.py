
import h5py
import numpy as np

with h5py.File('result_data/model_91_60_40_20/error_and_accuracy.h5','r') as hdf:
    #base_items = list(hdf.items())
    base_items = list(hdf.keys())
    print(base_items)
    input_data = hdf.get('training_loss')
    group_input_data = list(input_data.keys())
    print(group_input_data)
    d_1 = hdf.get('training_loss')
    print(d_1[:])



