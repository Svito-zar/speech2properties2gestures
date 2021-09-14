import numpy as np
import h5py
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
    joint_array = np.concatenate((timestamps, property_features), axis=1)
    joint_dataset = {"Audio":load_dataset("Audio"), "Text":load_dataset("Text"), "All":joint_array}
    # Gather indices of frames without a gesture
    zero_inds = np.where(np.max(property_features, axis=1) == 0)
    # save into npy
    property_dataset = np.save(join(DATASET_DIR, "All_properties.npy"), joint_array)
    # create hdf5 file
    save_dataset(join(DATASET_DIR, "all_together"), joint_dataset, zero_inds)
    print("Created joint property dataset with shape:", joint_array.shape)

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

    # Save gesture existance labels with the timestamps
    exist_dataset = {"Audio":load_dataset("Audio"), "Text":load_dataset("Text"), "exist":gesture_existance_array}
    # create hdf5 file
    print("Created joint property dataset with shape:", gesture_existance_array.shape)
    save_dataset(join(DATASET_DIR, "gest_exist"), exist_dataset)
    print("FINAL SHAPE:", gesture_existance_array.shape)
    print("# frames with gesture:", gesture_existance_array[:, 2].sum())
    
    return gesture_existance_array

def remove_zeros(gesture_existance_array):
    """
    Remove frames where none of the gesture properties are active.
    """
    print_banner("Removing frames without a gesture")
    output_dir = join(DATASET_DIR, "no_zeros")
    # Gather indices of frames without a gesture
    zero_inds = np.where(gesture_existance_array[:, 2] == 0)
    
    dataset_names = ["Phrase_properties", "Phase_properties", "Semantic_properties", "Audio", "Text"]
    datasets = { name : load_dataset(name) for name in dataset_names }

    # create hdf5 file
    save_dataset(output_dir, datasets, zero_inds)


def load_dataset(fname):
    return np.load(join(DATASET_DIR, fname + ".npy"))


def save_dataset(output_dir, datasets, zero_inds=False):
    if not isdir(output_dir):
        makedirs(output_dir)
    hf = h5py.File(output_dir + "/train_n_val.hdf5", 'a')  # open a hdf5 file
    g1 = hf.create_group('train')  # create group
    for dataset_name, array in datasets.items():
        if zero_inds != False:
            array = np.delete(array, zero_inds, axis=0)
        if str(dataset_name).find("properties") != -1:
            # remove `properties` from the name
            feat_name = dataset_name[:-11]
        else:
            feat_name = dataset_name
        g1.create_dataset(feat_name, data=array)
        print("\t", dataset_name, array.shape)
    print("Saved arrays to", abspath(output_dir))  
    hf.close()

def print_banner(text):
    print()
    print("-"*30)
    print(text)
    print("-"*30)

if __name__ == "__main__":
    DATASET_DIR = "/home/taras/Documents/storage/Saga/dataset/processed/numpy_arrays/train_n_val/"
    print("Using dataset dir:", abspath(DATASET_DIR))
    
    main()
