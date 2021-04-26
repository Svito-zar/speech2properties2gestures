from __future__ import print_function, division
from os import path
import torch
import numpy as np
from torch.utils.data import Dataset

torch.set_default_tensor_type('torch.FloatTensor')


class GesturePropDataset(Dataset):
    """Gesture Properties from the SAGA dataset."""

    def __init__(self, root_dir, dataset_type, features_name, indices_to_subsample = None):
        """
        Args:
            root_dir (string): Directory with the datasat.
        """
        self.root_dir = root_dir

        # define data file name
        self.type = dataset_type
        self.x_file_name = root_dir + dataset_type + "_X_" + features_name + ".npy"
        self.y_file_name = root_dir + dataset_type + "_Y_" + features_name + ".npy"

        # read dataset
        self.x_dataset = np.load(self.x_file_name)
        self.y_dataset = np.load(self.y_file_name)

        # select indices if provided
        if indices_to_subsample is not None:
            self.x_dataset = self.x_dataset[indices_to_subsample]
            self.y_dataset = self.y_dataset[indices_to_subsample]

        self.len = self.y_dataset.shape[0]


    def __len__(self):
        return self.len


    def __getitem__(self, idx):

        text = self.x_dataset[idx]
        property = self.y_dataset[idx]

        if len(text) == 0:
            raise Exception("Missing text!")

        sample = {'text': text, 'property': property}

        return sample


if __name__ == "__main__":

    # Test the dataset class
    gen_folder = "/home/tarask/Documents/Datasets/SaGa/Processed/feat/R_Phase_n_Practice/"
    dataset_name = subfolder = "test"
    feature_name = "gesture_phrase_n_practice_Right"

    TestDataset = GesturePropDataset(gen_folder, dataset_name, feature_name)