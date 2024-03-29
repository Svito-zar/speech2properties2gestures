from __future__ import print_function, division
from os import path
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

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

    def __init__(self, root_dir, scalers=None, train=True):
        """
        Args:
            root_dir (string): Directory with the datasat.
        """
        self.root_dir = root_dir
        # Get the data
        if train:
            audio = np.load(path.join(root_dir, 'X_train.npy')).astype(np.float32)
            #text = np.load(path.join(root_dir, 'T_train.npy')).astype(np.float32)
            gesture = np.load(path.join(root_dir, 'Y_train.npy')).astype(np.float32)
        else:
            audio = np.load(path.join(root_dir, 'X_dev.npy')).astype(np.float32)
            #text = np.load(path.join(root_dir, 'T_dev.npy')).astype(np.float32)
            gesture = np.load(path.join(root_dir, 'Y_dev.npy')).astype(np.float32)

        # upsample text to get the same sampling rate as the audio
        #cols = np.linspace(0, text.shape[1], endpoint=False, num=text.shape[1] * 2, dtype=int)
        #text = text[:, cols, :]

        self.audio_dim = audio.shape[-1]

        # Standartize
        if train:
            self.audio, audio_scaler = fit_and_standardize(audio)
            #self.text, text_scaler = fit_and_standardize(text)
            self.gesture, gesture_scaler = fit_and_standardize(gesture)
            self.scalers = [audio_scaler, gesture_scaler]
        else:
            [audio_scaler, gesture_scaler] = scalers
            self.audio  = standardize(audio, audio_scaler)
            #self.text  = standardize(text, text_scaler)
            self.gesture  = standardize(gesture, gesture_scaler)

        # ToDo: add Text and text_scaler

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        audio = self.audio[idx]
        gesture = self.gesture[idx]
        text = 0 # self.text[idx]

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