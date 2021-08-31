from os import path
from os.path import join
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset

torch.set_default_tensor_type('torch.FloatTensor')


class GesturePropDataset(Dataset):
    """Gesture Properties from the SAGA dataset."""

    def __init__(self, 
        property_name,
        speech_modality,
        root_dir = "../../dataset/processed/numpy_arrays", 
        dataset_type = "train_n_val/no_zeros",
        indices_to_subsample = None
    ):
        """
        Args:
            root_dir (string): Directory with the datasat.
        """
        self.file_name = join(root_dir, dataset_type, "train_n_val.hdf5")
        self.sp_mod = speech_modality

        # define the modality used
        self.speech_modality = speech_modality

        # define the property being modeled
        self.property = property_name

        # define indices
        prop_hdf5_obj = h5py.File(self.file_name, "r")["train"][self.property]
        size = prop_hdf5_obj.shape[0]
        self.indices = np.arange(size)

        # save recordings IDs
        self.record_ids = prop_hdf5_obj[:, 0]

        # Optional subsampling
        if indices_to_subsample is not None:
            self.indices = self.indices[indices_to_subsample]
    
        self.calculate_frequencies()


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        with h5py.File(self.file_name, "r") as data:

            def get_data_item(who):
                return data["train"].get(who)[index]

            gest_property = get_data_item(self.property)

            if len(gest_property) == 0:
                raise Exception("Missing datapoint!")

            if self.sp_mod == "text":
                text = get_data_item("Text")
                sample = {'text': text, 'property': gest_property}

            elif self.sp_mod == "audio":
                audio = get_data_item("Audio")
                sample = {'audio': audio, 'property': gest_property}

            elif self.sp_mod == "both":
                text = get_data_item("Text")
                audio = get_data_item("Audio")
                sample = {'audio': audio, 'text': text, 'property': gest_property}

        return sample


    def calculate_frequencies(self):
        property_dataset = h5py.File(self.file_name, "r")["train"][self.property]
        numb_feat = property_dataset.shape[1] - 2
        freq = np.zeros(numb_feat)
        for feat in range(numb_feat):
            column = property_dataset[:, 2 + feat]
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
