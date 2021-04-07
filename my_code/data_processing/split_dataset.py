"""
This script is used to split the dataset into train, test and dev sets.
More info on its usage is given in the main README.md file 

@authors: Taras Kucherenko, Rajmund Nagy
"""
import os
import shutil
from pathlib import Path
from os import path
from os.path import join

from my_code.data_processing.text_features.parse_json_transcript import encode_json_transcript_with_bert, encode_json_transcript_with_fasttext
from my_code.data_processing import tools
# Params
from my_code.data_processing.data_params import processing_argparser
from my_code.data_processing.utils import get_file_name, get_file_extension


def get_split_from_index(file_ind):
    """Return the dataset split ("dev" or "test" or "train") for the given file index."""

    if file_ind in [2, 13, 17]:
        return "test"
    elif file_ind in [7, 8]:
        return "dev"
    else:
        return "train"


def check_dataset_directories(raw_data_dir, proc_data_dir):
    """
    Do the following two sanity checks:
    
    1)  Verify that 'raw_data_dir' exists and that it contains the 
        'Audio', 'Transcripts' and 'Motion' subdirectories; 
    2)  Check if the 'proc_data_dir' is empty, if it is not, then print
    """
    # Ensure that proc_data_dir is either empty or that it can be deleted
    if path.isdir(proc_data_dir) and os.listdir(proc_data_dir):
        print(f"WARNING: Result directory '{proc_data_dir}' already exists!", end=' ')
        print("All files in this directory will be deleted!")
        print("\nType 'ok' to clear the directory, and anything else to abort the program.")

        if input() == 'ok':
            shutil.rmtree(proc_data_dir)
        else:
            exit(-1)

    # Ensure that raw_data_dir is correct
    if not path.isdir(raw_data_dir):
        abs_path = path.abspath(raw_data_dir)

        print(f"ERROR: The given dataset folder for the raw data ({abs_path}) does not exist!")
        print("Please, provide the correct path to the dataset in the `-raw_data_dir` argument.")
        exit(-1)

    speech_dir     = path.join(raw_data_dir, "Audio")
    transcript_dir = path.join(raw_data_dir, "Transcripts")
    motion_dir     = path.join(raw_data_dir, "Motion")

    # Ensure that raw_data_dir contains the right subfolders
    for sub_dir in [speech_dir, transcript_dir, motion_dir]:
        if not path.isdir(sub_dir):
            _, name = path.split(sub_dir)
            print(f"ERROR: The '{name}' directory is missing from the given dataset folder: '{raw_data_dir}'!") 
            exit(-1)

def _create_data_directories(processed_d_dir):
    """Create subdirectories for the dataset splits."""
    dir_names = ["dev", "test", "train"]
    sub_dir_names = ["inputs", "labels", "sequences"]

    os.makedirs(processed_d_dir, exist_ok = True)
    
    print("Creating the datasets in the following directories:") 
    for dir_name in dir_names:
        dir_path = path.join(processed_d_dir, dir_name)
        print('  ', path.abspath(dir_path))
        os.makedirs(dir_path, exist_ok=True)  # e.g. ../../dataset/processed/train

        for sub_dir_name in sub_dir_names:
            dir_path = path.join(processed_d_dir, dir_name, sub_dir_name)
            os.makedirs(dir_path, exist_ok = True) # e.g. ../../dataset/processed/train/inputs/
    print()

def create_dataset_splits_saga(raw_data_dir, proc_data_dir):    
    """
    Create the train/dev/test dataset splits in 'proc_data_dir' and move the raw files there.
    """
    _create_data_directories(proc_data_dir)

    for subfolder in sorted(os.listdir(raw_data_dir)):
        subfolder = join(raw_data_dir, subfolder)
        if os.path.isdir(subfolder):
            print("\n\n")
            for file in sorted(os.listdir(subfolder)):
                copy_saga_file(join(subfolder, file), proc_data_dir)

def copy_saga_file(file_path, proc_data_dir):
    """
    Copy the given raw data file to the corresponding data split, based on the 
    file index and the data type (.bvh for motion, .hdf5 for text, .mov.wav for audio).
    """
    file_name = get_file_name(file_path)
    file_extension = get_file_extension(file_path)

    if file_name == "":
        return

    print("File: ", file_name)
 
    # Remove the leading V from the file name to get the index
    file_ind = int(file_name[1:3])
    # "dev", "train" or "test"
    split = get_split_from_index(file_ind)
    
    if file_extension == "npz":
        target_dir = "labels"
    elif file_extension == "hdf5" or file_extension == "mov.wav":
        target_dir = "inputs"
    else:
        print(f"{file_name}.{file_extension} (skipped)")
        return
    
    target_dir = join(proc_data_dir, split, target_dir)
    
    print(f"{file_name}.{file_extension}  \t -> \t {target_dir}")

    shutil.copy(file_path, target_dir)

if __name__ == "__main__":
    args = processing_argparser.parse_args()
    
    check_dataset_directories(args.raw_data_dir, args.proc_data_dir)
    create_dataset_splits_saga(args.raw_data_dir, args.proc_data_dir)
