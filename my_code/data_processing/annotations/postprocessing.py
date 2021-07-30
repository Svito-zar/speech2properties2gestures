import numpy as np
from os.path import join, isdir, abspath
from os import makedirs

def main():
    """
    This script will do the following three things:
        1) Create an array that contains all 3 gesture properties ("All_properties.npy")
        2) Create binary gesture existance array ("Gest_exist_properties.npy")
        3) Create new dataset arrays where zero vectors are removes (saved into the "no_zeros" folder)
    """
    concatenate_gesture_properties()
    gest_exist_array = create_gest_exist_array()
    remove_zeros(gest_exist_array)
    
def concatenate_gesture_properties():
    """
    Create and save an array that contains all 3 gesture properties.
    """
    print_banner("Creating joint property array")
    # Open the datasets
    dataset_names = ["Phrase_properties", "Phase_properties", "Semantic_properties", "Audio", "Text"]
    datasets = { name : load_dataset(name) for name in dataset_names }
    
    # Make sure that all lengths are equal
    lengths = [len(dataset) for dataset in datasets.values()]
    assert len(np.unique(lengths)) == 1
        
    # Make sure that the timestamps match
    timestamps_per_array = [dataset[:, :2] for idx, (name, dataset) in enumerate(datasets.items()) if "properties" in name]
    timestamps_per_array = np.stack(timestamps_per_array)
    assert len(np.unique(timestamps_per_array, axis=0)) == 1

    # Stack all 3 properties in each frame
    timestamps = datasets["Phase_properties"][:, :2]
    property_features = np.concatenate(
        [dataset[:, 2:] for name, dataset in datasets.items() if "properties" in name],
        axis = 1
    )

    # Save property labels with the timestamps
    joint_dataset = np.concatenate((timestamps, property_features), axis=1)
    np.save(join(DATASET_DIR, "All_properties.npy"), joint_dataset)
    print("Created joint property dataset with shape:", joint_dataset.shape)

    return joint_dataset

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

    print(f"Number of labels: Phase: {has_phase}, Phrase: {has_phrase}, Semantic: {has_semantic}, No feat: {no_feat}")

def create_gest_exist_array():
    """
    Create binary gesture existance array, which contains value 1 when at least
    one property is nonzero, and value 0 otherwise.
    """
    print_banner("Creating gest. exist. array")
    property_dataset = np.load(join(DATASET_DIR, "All_properties.npy"))
    print_stats(property_dataset)

    gesture_existance_array = np.zeros((property_dataset.shape[0], 3))
    # Copy timestamps
    gesture_existance_array[:, :2] = property_dataset[:, :2]

    # Set binary value for gesture existance
    for ind, feat in enumerate(property_dataset):
        gesture_existance_array[ind, 2] = np.max(feat[2:])

    assert gesture_existance_array[:, 2].max() == 1 
    assert gesture_existance_array[:, 2].min() == 0

    np.save(join(DATASET_DIR, "Gest_exist_properties.npy"), gesture_existance_array)
    print("FINAL SHAPE:", gesture_existance_array.shape)
    print("# frames with gesture:", gesture_existance_array[:, 2].sum())
    
    return gesture_existance_array

def remove_zeros(gesture_existance_array):
    """
    Remove frames where none of the gesture properties are active.
    """
    print_banner("Removing frames without a gesture")
    output_dir = join(DATASET_DIR, "no_zeros")
    if not isdir(output_dir):
        makedirs(output_dir)
    # Gather indices of frames without a gesture
    zero_inds = np.where(gesture_existance_array[:, 2] == 0)
    
    dataset_names = ["Phrase_properties", "Phase_properties", "Semantic_properties", "Audio", "Text"]
    datasets = { name : load_dataset(name) for name in dataset_names }
    
    print("Final dataset shapes:")
    for dataset_name, array in datasets.items():
        no_zero_array = np.delete(array, zero_inds, axis=0)
        np.save(join(output_dir, dataset_name), no_zero_array)
        print("\t", dataset_name, no_zero_array.shape)
    print("Saved arrays to", abspath(output_dir))  
    
def load_dataset(fname):
    return np.load(join(DATASET_DIR, fname + ".npy"))

def save_dataset(fname, array):
    np.save(join(DATASET_DIR, fname), array)

def print_banner(text):
    print()
    print("-"*30)
    print(text)
    print("-"*30)

if __name__ == "__main__":
    DATASET_DIR = "../../../dataset/processed/numpy_arrays/"
    print("Using dataset dir:", abspath(DATASET_DIR))
    
    main()