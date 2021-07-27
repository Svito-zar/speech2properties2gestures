import numpy as np
from os.path import join

from torch.utils import data

def print_stats(dataset):
    no_feat = 0
    has_phase = 0
    has_phrase = 0
    has_semantic = 0
    for feat in dataset:
        if np.sum(feat[2:]) == 0:
            no_feat += 1
        
        if np.sum(feat[2:6]) > 0:
            has_phrase += 1
        
        if np.sum(feat[6:11]) > 0:
            has_phase += 1

        if np.sum(feat[11:]) > 0:
            has_semantic += 1

    print(f"Phase: {has_phase}, Phrase: {has_phrase}, Semantic: {has_semantic}, No feat: {no_feat}")


dataset_dir = "../../../dataset/processed/numpy_arrays/"

property_dataset = np.load(join(dataset_dir, "All_properties.npy"))

print_stats(property_dataset)

gesture_existance_array = np.zeros((property_dataset.shape[0], 3))
# Copy timestamps
gesture_existance_array[:, :2] = property_dataset[:, :2]

# Set binary value for gesture existance
for ind, feat in enumerate(property_dataset):
    gesture_existance_array[ind, 2] = np.max(feat[2:])

assert gesture_existance_array[:, 2].max() == 1 
assert gesture_existance_array[:, 2].min() == 0

np.save(join(dataset_dir, "Gest_exist_properties.npy"), gesture_existance_array)
print("FINAL SHAPE:", gesture_existance_array.shape)
print("# gestures:", gesture_existance_array[:, 2].sum())