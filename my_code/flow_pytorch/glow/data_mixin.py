import h5py
from pathlib import Path
import torch
from flow_pytorch.glow_mimicry_dataset import GlowMimicryDataset
from torch.utils.data.dataloader import DataLoader


class DataMixin:
    
    def _data_loader(self, file_name, shuffle=True, seq_len=25):

        dataset = GlowMimicryDataset(
            Path(self.hparams.dataset_root) / file_name,
            seq_len=seq_len,
            data_hparams=self.hparams.Data,
            conditioning_hparams=self.hparams.Conditioning,
        )

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_dataloader_workers,
            shuffle=shuffle,
            drop_last=False,
        )

    def train_dataloader(self):
        return self._data_loader(
            self.hparams.Train["data_file_name"], seq_len=self.hparams.Train["seq_len"],
        )

    def val_dataloader(self):
        return self._data_loader(
            self.hparams.Validation["data_file_name"],
            shuffle=False,
            seq_len=self.hparams.Validation["seq_len"],
        )

    def test_dataloader(self):
        return self._data_loader(
            self.hparams.Test["data_file_name"],
            shuffle=False,
            seq_len=self.hparams.Test["seq_len"],
        )
