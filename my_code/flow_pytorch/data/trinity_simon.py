"""
This file contain the Trinity class for working with
the Trinity Speech-Gesture Dataset

author: Simon Alexanderson
"""

import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


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


class MotionDataset(Dataset):
    """
    Motion dataset.
    Prepares conditioning information (previous poses + control signal) and the corresponding next poses"""

    def __init__(self, control_data, joint_data, seqlen, n_lookahead, dropout):
        """
        Args:
        control_data: The control input
        joint_data: body pose input
        Both with shape (samples, time-slices, features)
        seqlen: number of autoregressive body poses and previous control values
        n_lookahead: number of future control-values
        dropout: (0-1) dropout probability for previous poses
        """
        self.seqlen = seqlen
        self.dropout = dropout
        seqlen_control = seqlen + n_lookahead + 1

        # For LSTM network
        n_frames = joint_data.shape[1]

        # Joint positions for n previous frames
        autoreg = self.concat_sequence(self.seqlen, joint_data[:, :n_frames - n_lookahead - 1, :])

        # Control for n previous frames + current frame
        control = self.concat_sequence(seqlen_control, control_data)

        # conditioning

        print("autoreg:" + str(autoreg.shape))
        print("control:" + str(control.shape))
        new_cond = np.concatenate((autoreg, control), axis=2)

        # joint positions for the current frame
        x_start = seqlen
        new_x = self.concat_sequence(1, joint_data[:, x_start:n_frames - n_lookahead, :])
        self.x = new_x
        self.cond = new_cond

        # TODO TEMP swap C and T axis to match existing implementation
        self.x = np.swapaxes(self.x, 1, 2)
        self.cond = np.swapaxes(self.cond, 1, 2)

        print("self.x:" + str(self.x.shape))
        print("self.cond:" + str(self.cond.shape))

    def n_channels(self):
        return self.x.shape[1], self.cond.shape[1]

    def concat_sequence(self, seqlen, data):
        """
        Concatenates a sequence of features to one.
        """
        nn, n_timesteps, n_feats = data.shape
        L = n_timesteps - (seqlen - 1)
        inds = np.zeros((L, seqlen)).astype(int)

        # create indices for the sequences we want
        rng = np.arange(0, n_timesteps)
        for ii in range(0, seqlen):
            inds[:, ii] = np.transpose(rng[ii:(n_timesteps - (seqlen - ii - 1))])

            # slice each sample into L sequences and store as new samples
        cc = data[:, inds, :].copy()

        # print ("cc: " + str(cc.shape))

        # reshape all timesteps and features into one dimention per sample
        dd = cc.reshape((nn, L, seqlen * n_feats))
        # print ("dd: " + str(dd.shape))
        return dd

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        Returns poses and conditioning.
        If data-dropout sould be applied, a random selection of the previous poses is masked.
        The control is not masked
        """

        if self.dropout > 0.:
            n_feats, tt = self.x[idx, :, :].shape
            cond_masked = self.cond[idx, :, :].copy()

            keep_pose = np.random.rand(self.seqlen, tt) < (1 - self.dropout)

            # print(keep_pose)
            n_cond = cond_masked.shape[0] - (n_feats * self.seqlen)
            mask_cond = np.full((n_cond, tt), True)

            mask = np.repeat(keep_pose, n_feats, axis=0)
            mask = np.concatenate((mask, mask_cond), axis=0)
            # print(mask)

            cond_masked = cond_masked * mask
            sample = {'x': self.x[idx, :, :], 'cond': cond_masked}
        else:
            sample = {'x': self.x[idx, :, :], 'cond': self.cond[idx, :, :]}

        return sample


class TestDataset(Dataset):
    """Test dataset."""

    def __init__(self, control_data, joint_data):
        """
        Args:
        control_data: The control input
        joint_data: body pose input
        Both with shape (samples, time-slices, features)
        """
        # Joint positions
        self.autoreg = joint_data

        # Control
        self.control = control_data

    def __len__(self):
        return self.autoreg.shape[0]

    def __getitem__(self, idx):
        sample = {'autoreg': self.autoreg[idx, :], 'control': self.control[idx, :]}
        return sample


class TrinityDataset():

    def __init__(self, hparams, data_dir):
        data_root = data_dir

        # load data
        train_input = np.load(os.path.join(data_root, 'train_input_' + str(hparams.Data.framerate) + 'fps.npz'))[
            'clips'].astype(np.float32)
        train_output = np.load(os.path.join(data_root, 'train_output_' + str(hparams.Data.framerate) + 'fps.npz'))[
            'clips'].astype(np.float32)
        val_input = np.load(os.path.join(data_root, 'val_input_' + str(hparams.Data.framerate) + 'fps.npz'))[
            'clips'].astype(np.float32)
        val_output = np.load(os.path.join(data_root, 'val_output_' + str(hparams.Data.framerate) + 'fps.npz'))[
            'clips'].astype(np.float32)

        # use this to generate visualizations for network tuning. It contains the same data as val_input, but sliced into longer 20-sec exerpts
        test_input = np.load(os.path.join(data_root, 'dev_input_' + str(hparams.Data.framerate) + 'fps.npz'))[
            'clips'].astype(np.float32)

        # make sure the test data is at least one batch size
        self.n_test = test_input.shape[0]
        n_tiles = 1 + hparams.Train.batch_size // self.n_test
        test_input = np.tile(test_input.copy(), (n_tiles, 1, 1))

        # Standartize
        train_input, input_scaler = fit_and_standardize(train_input)
        train_output, output_scaler = fit_and_standardize(train_output)
        val_input = standardize(val_input, input_scaler)
        val_output = standardize(val_output, output_scaler)
        test_input = standardize(test_input, input_scaler)
        test_output = np.zeros((test_input.shape[0], test_input.shape[1], train_output.shape[2])).astype(np.float32)

        # Create pytorch data sets
        self.train_dataset = MotionDataset(train_input, train_output, hparams.Data.seqlen, hparams.Data.n_lookahead,
                                           hparams.Data.dropout)
        self.validation_dataset = MotionDataset(val_input, val_output, hparams.Data.seqlen, hparams.Data.n_lookahead,
                                                hparams.Data.dropout)
        self.test_dataset = TestDataset(test_input, test_output)

        # Store scaler
        self.scaler = output_scaler

    def save_animation(self, motion_data, filename):
        anim_clips = inv_standardize(motion_data[:(self.n_test), :, :], self.scaler)
        np.savez(filename + ".npz", clips=anim_clips)

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_validation_dataset(self):
        return self.validation_dataset