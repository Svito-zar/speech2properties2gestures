from tqdm import tqdm
import numpy as np
import h5py
from os.path import join
import os
import pickle

reference_folder = "/tmp/mozilla_work0/SaGa_feat_split/train_n_val"
folder_to_test = "feat/"
dict_file = "dict.pkl"

with open(dict_file, 'rb') as handle:
    PROPERTY_DICT = pickle.load(handle)
# ----------------------------------------

def print_numpy_array(array, key):
    try:
        n_columns = array.shape[1]
    except:
        array = array[np.newaxis, :]
        n_columns = array.shape[1]
    row_format = "{:<10}" * n_columns
    
    header = ["start", "end"] + list(PROPERTY_DICT[key].values())
    print("\t", row_format.format(*[label[:8] for label in header]))
    for row in array:
        print("\t", row_format.format(*row))

all_good = False
for file in sorted(os.listdir(reference_folder)):
    if "text" in file:
        continue

    if all_good:
        print("----> COMPLETE MATCH")
    all_good = True
    print("\n", file)

    tested_file = h5py.File(join(folder_to_test, file), "r")
    reference_file = h5py.File(join(reference_folder, file), "r")

    assert list(tested_file.keys()) == list(reference_file.keys())
    
    for key in list(tested_file.keys()):
        tested_data = np.asarray(tested_file.get(key))
        reference_data = np.asarray(reference_file.get(key))
         
        if not np.array_equal(tested_data, reference_data):
            all_good = False
            print("---->", key, "MISMATCH", end="")
            if tested_data.shape == reference_data.shape:
                print("\t(but shape is equal)")
            else:
                print("\t(inequal shapes: {} -> {})".format(reference_data.shape, tested_data.shape))

            tested_data_set = {tuple(row) for row in tested_data}
            reference_data_set = {tuple(row) for row in reference_data}

            print("[Extra rows]")
            extra_rows = np.asarray(sorted(tested_data_set - reference_data_set))
            print_numpy_array(extra_rows, key)
            
            print("[Missing rows]")
            missing_rows = np.asarray(sorted(reference_data_set - tested_data_set))
            print_numpy_array(missing_rows, key)
            