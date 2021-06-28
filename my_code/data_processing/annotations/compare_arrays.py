import numpy as np 

old = np.load("../../../dataset/processed/numpy_arrays/train_n_val_Y_Phrase.npy")
new = np.load("../../../dataset/processed/numpy_arrays/Phrase_properties.npy")


# compare speakers
speakers_old = np.unique(old[:, 0], axis=0)
speakers_new = np.unique(new[:, 0], axis=0)

assert np.array_equal(speakers_old, speakers_new)

for speaker in speakers_old:
    speaker_data_old = old[np.argwhere(old[:, 0] == speaker), 0]
    speaker_data_new = new[np.argwhere(new[:, 0] == speaker), 0]

    print(speaker_data_old.shape, "->", speaker_data_new.shape)