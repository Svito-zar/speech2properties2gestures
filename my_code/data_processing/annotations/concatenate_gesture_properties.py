import numpy as np
from os.path import join

dataset_dir = "../../../dataset/processed/numpy_arrays/train_n_val/"

get_dataset = lambda fname : np.load(join(dataset_dir, fname + ".npy"))

dataset_names = ["Phrase_properties", "Phase_properties", "Semantic_properties", "Audio", "Text"]
datasets = { name : get_dataset(name) for name in dataset_names }

# Make sure that all lengths are equal
lengths = [len(dataset) for dataset in datasets.values()]
assert len(np.unique(lengths)) == 1

# Make sure that the timestamps match
timestamps_per_array = [dataset[:, :2] for idx, (name, dataset) in enumerate(datasets.items()) if "properties" in name]
timestamps_per_array = np.stack(timestamps_per_array)
assert len(np.unique(timestamps_per_array, axis=0)) == 1

timestamps = datasets["Phase_properties"][:, :2]
property_features = np.concatenate(
    [dataset[:, 2:] for name, dataset in datasets.items() if "properties" in name],
    axis = 1
)

joint_dataset = np.concatenate((timestamps, property_features), axis=1)

print("FINAL SHAPE:", joint_dataset.shape)
np.save(join(dataset_dir, "All_properties.npy"), joint_dataset)