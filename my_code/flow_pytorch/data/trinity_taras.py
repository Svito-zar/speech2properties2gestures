from __future__ import print_function, division
from os import path
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import h5py
import random

torch.set_default_tensor_type('torch.FloatTensor')


def fit_and_standardize(data):
    shape = data.shape
    flat = data.copy().reshape((shape[0] * shape[1], shape[2]))
    scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat).reshape(shape)
    return scaled, scaler


def standardize(data, scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0] * shape[1], shape[2]))
    scaled = scaler.transform(flat).reshape(shape)
    return scaled


def inv_standardize(data, scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0] * shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled


class SpeechGestureDataset(Dataset):
    """Trinity Speech-Gesture Dataset class."""

    def __init__(self, root_dir, train=True):
        """
        Args:
            root_dir (string): Directory with the datasat.
        """
        self.root_dir = root_dir

        # define data file name
        if train:
            self.file_name = path.join(root_dir, 'train.hdf5')
            self.type = "train"
        else:
            self.file_name  = path.join(root_dir, 'dev.hdf5')
            self.type = "dev"

        scalers_file = root_dir + "/scalers.npy"
        self.scalers = np.load(scalers_file, allow_pickle=True)

        # Define dataset size
        with h5py.File(self.file_name, "r") as data:
            audio_data =  data[self.type]["audio"]
            self.len = len(audio_data)



    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        with h5py.File(self.file_name, "r") as data:
            audio = data[self.type]["audio"][idx]
            text = data[self.type]["text"][idx]
            gesture = data[self.type]["gesture"][idx]

            # upsample text to get the same sampling rate as the audio
            cols = np.linspace(0, text.shape[0], endpoint=False, num=text.shape[0] * 2, dtype=int)
            text = text[cols, :]

        if len(text) == 0:
            raise Exception("Missing text!")

        sample = {'audio': audio, 'gesture': gesture, 'text': text}

        return sample

    def get_scalers(self):
        return self.scalers


class ValidationDataset(Dataset):
    """Validation samples from the Trinity Speech-Gesture Dataset."""

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with the datasat.
        """
        self.root_dir = root_dir
        # Get the data
        self.audio = np.load(path.join(root_dir, 'dev_inputs', 'X_dev_NaturalTalking_01.npy')).astype(np.float32)
        self.text = np.load(path.join(root_dir, 'dev_inputs', 'T_dev_NaturalTalking_01.npy')).astype(np.float32)
        # upsample text to get the same sampling rate as the audio
        cols = np.linspace(0, self.text.shape[0], endpoint=False, num=self.text.shape[0] * 2, dtype=int)
        self.text = self.text[cols, :]

        self.start_times = [99.9, 140.6, 164.3, 257.7, 269.6, 278.9, 315.8, 372.5, 476.9]
        self.end_times = [104.7, 149.5, 166, 259.6, 272.1, 288.2, 317.9, 377.3, 481.6]

        self.audio_dim = self[0]['audio'].shape[-1]

    def __len__(self):
        return len(self.start_times)

    def __getitem__(self, idx):
        start = int(self.start_times[idx] * 20)  # 20fps
        end = int(self.end_times[idx] * 20)  # 20fps

        audio = self.audio[start - 10:end + 20]
        text = self.text[start - 10:end + 20]

        sample = {'audio': audio, 'text': text}

        return sample