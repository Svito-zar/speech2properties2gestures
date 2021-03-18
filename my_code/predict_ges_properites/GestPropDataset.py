from __future__ import print_function, division
from os import path
import torch
import numpy as np
from torch.utils.data import Dataset

torch.set_default_tensor_type('torch.FloatTensor')


class GesturePropDataset(Dataset):
    """Gesture Properties from the SAGA dataset."""

    def __init__(self, root_dir, dataset_type, features_name):
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
        # convert binary encoding to labels for StratifiedKFold split
        y_labels = [int("".join(str(int(i)) for i in bin_list[2:]),2) for bin_list in self.y_dataset]
        self.y_labels =  np.array(y_labels)

        self.len = self.y_dataset.shape[0]

        self.calculate_frequencies()


    def __len__(self):
        return self.len


    def __getitem__(self, idx):

        text = self.x_dataset[idx]
        property = self.y_dataset[idx]

        if len(text) == 0:
            raise Exception("Missing text!")

        sample = {'text': text, 'property': property}

        return sample


    def calculate_frequencies(self):
        numb_feat = self.y_dataset.shape[1] - 2
        freq = np.zeros(numb_feat)
        for feat in range(numb_feat):
            column = self.y_dataset[:, 2 + feat]
            freq[feat] = np.sum(column)
            if freq[feat] < 30:
                freq[feat] = 10000
        self.class_freq = freq


    def get_freq(self):
        return self.class_freq


if __name__ == "__main__":

    # Test the dataset class
    gen_folder = "/home/tarask/Documents/Datasets/SaGa/Processed/feat/R_Phase_n_Practice/"
    dataset_name = subfolder = "test"
    feature_name = "gesture_phrase_n_practice_Right"

    TestDataset = GesturePropDataset(gen_folder, dataset_name, feature_name)
