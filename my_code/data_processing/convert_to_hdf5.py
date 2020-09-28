import numpy as np
from os import path
import h5py

from sklearn.preprocessing import StandardScaler

root_dir = "/home/taras/Documents/Datasets/SpeechToMotion/Irish/processed/GestureFlow_hdf5"

"""
hf = h5py.File(root_dir+"/dev.hdf5", 'r')
print(hf.keys())

n1 = hf.get('dev')

n1 = np.array(n1)
print(n1.shape)

group2 = hf.get('dev/gesture')
print(np.array(group2).shape)

exit()
"""



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


train = False

# Standartize
if train:

    # read the data
    audio = np.load(path.join(root_dir, 'X_train.npy')).astype(np.float32)
    text = np.load(path.join(root_dir, 'T_train.npy')).astype(np.float32)
    gesture = np.load(path.join(root_dir, 'Y_train.npy')).astype(np.float32)

    # scale it
    audio, audio_scaler = fit_and_standardize(audio)
    text, text_scaler = fit_and_standardize(text)
    gesture, gesture_scaler = fit_and_standardize(gesture)
    scalers = [audio_scaler, text_scaler, gesture_scaler]

    # save scalers
    scalers_file = root_dir + "/scalers.npy"
    np.save(scalers_file, scalers)

    # create hdf5 file
    hf = h5py.File(root_dir + "/train.hdf5", 'a')  # open a hdf5 file

    g1 = hf.create_group('train')  # create group

else:

    # read the data
    audio = np.load(path.join(root_dir, 'X_dev.npy')).astype(np.float32)
    text = np.load(path.join(root_dir, 'T_dev.npy')).astype(np.float32)
    gesture = np.load(path.join(root_dir, 'Y_dev.npy')).astype(np.float32)

    # read scallers
    # save scalers
    scalers_file = root_dir + "/scalers.npy"
    scalers = np.load(scalers_file, allow_pickle=True)

    # scale the data
    [audio_scaler, text_scaler, gesture_scaler] = scalers
    audio = standardize(audio, audio_scaler)
    text = standardize(text, text_scaler)
    gesture = standardize(gesture, gesture_scaler)


    # create hdf5 file
    hf = h5py.File(root_dir + "/dev.hdf5", 'a')  # open a hdf5 file

    g1 = hf.create_group('dev')  # create group


g1.create_dataset('audio', data=audio)
g1.create_dataset('text', data=text)
g1.create_dataset('gesture', data=gesture)

print(g1.get('dev').shape)
# write the data to hdf5 file
hf.close()  # close the hdf5 file