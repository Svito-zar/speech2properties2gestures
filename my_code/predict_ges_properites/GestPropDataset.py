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

        # define the modality used
        self.sp_mod = speech_modality

        # define the property being modeled
        self.property = property_name

        # define indices
        dataset_obj = h5py.File(self.file_name, "r")
        prop_hdf5_obj = dataset_obj["train"][self.property]
        size = prop_hdf5_obj.shape[0]
        self.indices = np.arange(size)

        # ensure that the data dims match
        assert len(dataset_obj["train"]["Audio"]) == len(dataset_obj["train"]["Text"])
        assert len(dataset_obj["train"]["Audio"]) == len(prop_hdf5_obj)

        # save recordings IDs
        self.record_ids = prop_hdf5_obj[:, 0]

        # Optional subsampling
        if indices_to_subsample is not None:
            self.indices = self.indices[indices_to_subsample]
    
        self.calculate_frequencies(dataset_obj)

        dataset_obj.close()


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        with h5py.File(self.file_name, "r") as data:

            def get_data_item(modality):
                return data["train"].get(modality)[index]

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


    def calculate_frequencies(self, prop_hdf5_obj):
        property_dataset = prop_hdf5_obj["train"][self.property]
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
