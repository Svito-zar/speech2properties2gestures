"""
This file is doing the opposite of the merging script
https://github.com/Svito-zar/probabilistic-gesticulator/blob/conv_seq_model/my_code/data_processing/annotations/merge_feat_together.py
Typically it is being applied after zeros have been removed
"""

import numpy as np


def save_dataset(gen_folder, feature_array, A_array, X_array, feature_folder,  feature_name):
    """
    Save a dataset for a given gesture property
    Args:
        gen_folder:       general folder where the processed dataset is lying
        feature_array:    array with the current feature [T, D]
        A_array:          array with the audio data [T, 11, D]
        X_array:          array with the text data [T, 7, D]
        feature_folder:   folder where this feature data should be stored
        feature_name:     name of the current feature

    Returns:
        nothing, saves files in a given folder

    """
    np.save(gen_folder + feature_folder + '/train_n_val_Y_'+feature_name+'.npy', feature_array)
    np.save(gen_folder + feature_folder + '/train_n_val_A_'+feature_name+'.npy', A_array)
    np.save(gen_folder + feature_folder + '/train_n_val_X_'+feature_name+'.npy', X_array)


gen_folder = "/home/tarask/Documents/Datasets/SaGa/Processed/feat/Text_n_AudioBased/"

X_file_name = gen_folder + "AllTogether/train_n_val_X_all.npy"
X = np.load(X_file_name, allow_pickle=True)
print("Text dataset shape: ", X.shape)

A_file_name = gen_folder + "AllTogether/train_n_val_A_all.npy"
A = np.load(A_file_name, allow_pickle=True)
print("Audio dataset shape: ", A.shape)

Y_file_name = gen_folder + "AllTogether/train_n_val_Y_all.npy"
Y_all = np.load(Y_file_name, allow_pickle=True)
print("Features dataset shape: ", Y_all.shape)

# Separate each feature from all the features
timing = Y_all[:, :2]

# Speech Semant
Y_s_semant_w_timings = Y_all[:, :8]
print(Y_s_semant_w_timings.shape)
### Save new files
save_dataset(gen_folder, Y_s_semant_w_timings, A, X, 'S_Semantic',  'R.S.Semantic Feature')

# Ges Semant
Y_g_semant = Y_all[:, 8:12]
Y_g_semant_w_timings = np.concatenate((timing, Y_g_semant), axis=1)
print(Y_g_semant_w_timings.shape)
### Save new files
save_dataset(gen_folder, Y_g_semant_w_timings, A, X, 'G_Semantic',  'Semantic')

# Ges Phrase
Y_g_phrase = Y_all[:, 12:16]
Y_g_phrase_w_timings = np.concatenate((timing, Y_g_phrase), axis=1)
print(Y_g_phrase_w_timings.shape)
### Save new files
save_dataset(gen_folder, Y_g_phrase_w_timings, A, X, 'Phrase',  'Phrase')

# Ges Phase
Y_g_phase = Y_all[:, 16:]
Y_g_phase_w_timings = np.concatenate((timing, Y_g_phase), axis=1)
print(Y_g_phase_w_timings.shape)
### Save new files
save_dataset(gen_folder, Y_g_phase_w_timings, A, X, 'Phase',  'Phase')