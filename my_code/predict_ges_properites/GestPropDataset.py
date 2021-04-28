from __future__ import print_function, division
from os import path
import torch
import numpy as np
from torch.utils.data import Dataset

torch.set_default_tensor_type('torch.FloatTensor')


class GesturePropDataset(Dataset):
    """Gesture Properties from the SAGA dataset."""

    def __init__(self, root_dir, dataset_type, features_name, speech_modality, indices_to_subsample = None):
        """
        Args:
            root_dir (string): Directory with the datasat.
        """
        self.root_dir = root_dir
        self.sp_mod = speech_modality

        # define data file name
        self.type = dataset_type
        self.y_file_name = root_dir + dataset_type + "_Y_" + features_name + ".npy"

        if speech_modality == "text":
            self.t_file_name = root_dir + dataset_type + "_X_" + features_name + ".npy"
            self.t_dataset = np.load(self.t_file_name)
            self.a_dataset = None
        elif speech_modality == "audio":
            self.a_file_name = root_dir + dataset_type + "_A_" + features_name + ".npy"
            self.a_dataset = np.load(self.a_file_name)
            self.t_dataset = None
        elif speech_modality == "both":
            self.t_file_name = root_dir + dataset_type + "_X_" + features_name + ".npy"
            self.a_file_name = root_dir + dataset_type + "_A_" + features_name + ".npy"
            self.t_dataset = np.load(self.t_file_name)
            self.a_dataset = np.load(self.a_file_name)
        else:
            raise TypeError("Unknown speech modality - " + speech_modality)

        # read prop dataset
        self.y_dataset = np.load(self.y_file_name)

        # select indices if provided
        if indices_to_subsample is not None:
            self.y_dataset = self.y_dataset[indices_to_subsample]
            if self.a_dataset is not None:
                self.a_dataset = self.a_dataset[indices_to_subsample]
            if self.t_dataset is not None:
                self.t_dataset = self.t_dataset[indices_to_subsample]

        self.len = self.y_dataset.shape[0]

        self.calculate_frequencies()


    def __len__(self):
        return self.len


    def __getitem__(self, idx):

        property = self.y_dataset[idx]

        if self.sp_mod == "text":
            text = self.t_dataset[idx]
            sample = {'text': text, 'property': property}
        elif self.sp_mod == "audio":
            audio = self.a_dataset[idx]
            sample = {'audio': audio, 'property': property}
        elif self.sp_mod == "both":
            text = self.t_dataset[idx]
            audio = self.a_dataset[idx]
            sample = {'audio': audio, 'text': text, 'property': property}

        if len(property) == 0:
            raise Exception("Missing datapoint!")

        return sample


    def calculate_frequencies(self):
        numb_feat = self.y_dataset.shape[1] - 2
        freq = np.zeros(numb_feat)
        for feat in range(numb_feat):
            column = self.y_dataset[:, 2 + feat]
            freq[feat] = np.sum(column)
            if freq[feat] < 50:
                freq[feat] = 1000
        self.class_freq = freq


    def get_freq(self):
        return self.class_freq


if __name__ == "__main__":

    # Test the dataset class
    gen_folder = "/home/tarask/Documents/Datasets/SaGa/Processed/feat/R_Phase_n_Practice/"
    dataset_name = subfolder = "test"
    feature_name = "gesture_phrase_n_practice_Right"

    TestDataset = GesturePropDataset(gen_folder, dataset_name, feature_name)
