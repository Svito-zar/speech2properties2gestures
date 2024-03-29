from __future__ import print_function, division
from os.path import join
import torch
import numpy as np
from torch.utils.data import Dataset

torch.set_default_tensor_type('torch.FloatTensor')


class GesturePropDataset(Dataset):
    """Gesture Properties from the SAGA dataset."""

    def __init__(self, 
        property_name,
        speech_modality,
        root_dir = "../../dataset/processed/numpy_arrays", 
        dataset_type = "train_n_val", 
        indices_to_subsample = None
    ):
        """
        Args:
            root_dir (string): Directory with the datasat.
        """
        get_data_file = lambda fname : np.load(join(root_dir, dataset_type, fname))
        self.sp_mod = speech_modality
        # Load speech input data
        if speech_modality == "text":
            self.text_dataset = get_data_file("Text.npy")
            self.audio_dataset = None
        
        elif speech_modality == "audio":
            self.text_dataset = None
            self.audio_dataset = get_data_file("Audio.npy")
        
        elif speech_modality == "both":
            self.audio_dataset = get_data_file("Audio.npy")
            self.text_dataset = get_data_file("Text.npy")
        
        else:
            raise TypeError("Unknown speech modality - " + speech_modality)

        # Load gesture property data
        self.property_dataset = get_data_file(f"{property_name}_properties.npy")
        
        # Optional subsampling
        if indices_to_subsample is not None:
            self.audio_dataset    = self.audio_dataset[indices_to_subsample]
            self.text_dataset     = self.text_dataset[indices_to_subsample]
            self.property_dataset = self.property_dataset[indices_to_subsample]

        assert len(self.property_dataset) == len(self.audio_dataset) == len(self.text_dataset)
        
        self.calculate_frequencies()

    def __len__(self):
        return len(self.property_dataset)


    def __getitem__(self, idx):
        gest_property = self.property_dataset[idx]

        if self.sp_mod == "text":
            text = self.text_dataset[idx]
            sample = {'text': text, 'property': gest_property}

        elif self.sp_mod == "audio":
            audio = self.audio_dataset[idx]
            sample = {'audio': audio, 'property': gest_property}

        elif self.sp_mod == "both":
            text = self.text_dataset[idx]
            audio = self.audio_dataset[idx]
            sample = {'audio': audio, 'text': text, 'property': gest_property}

        if len(gest_property) == 0:
            raise Exception("Missing datapoint!")

        return sample


    def calculate_frequencies(self):
        numb_feat = self.property_dataset.shape[1] - 2
        freq = np.zeros(numb_feat)
        for feat in range(numb_feat):
            column = self.property_dataset[:, 2 + feat]
            freq[feat] = np.sum(column)
            # TODO(RN) remove this?
            if freq[feat] < 50:
                freq[feat] = 1000
        self.class_freq = freq


    def get_freq(self):
        return self.class_freq


if __name__ == "__main__":
    TestDataset = GesturePropDataset("Gest_exist", "both")
    print(len(TestDataset), TestDataset.text_dataset.shape, TestDataset.audio_dataset.shape, TestDataset.property_dataset.shape)
