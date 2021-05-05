import numpy as np


def obtain_data_for_feature(gen_folder, prop):
    file_name = gen_folder + "train_n_val_Y_" + prop + ".npy"
    Y_train_n_val = np.load(file_name, allow_pickle=True)

    file_name = gen_folder + "train_n_val_X_" + prop + ".npy"
    X_train_n_val = np.load(file_name, allow_pickle=True)

    file_name = gen_folder + "train_n_val_A_" + prop + ".npy"
    A_train_n_val = np.load(file_name, allow_pickle=True)

    print(prop + " shape : ", Y_train_n_val.shape)

    return A_train_n_val, X_train_n_val, Y_train_n_val


gen_folder = "/home/tarask/Documents/Datasets/SaGa/Processed/feat/Text_n_AudioBased/"

# Read all the features
A_phrase, X_phrase, Y_phrase = obtain_data_for_feature(gen_folder+ "Phrase/", "Phrase")
A_phase, X_phase, Y_phase = obtain_data_for_feature(gen_folder+ "Phase/", "Phase")
A_g_semant, X_g_semant, Y_g_semant = obtain_data_for_feature(gen_folder+'G_Semantic/', "Semantic")
A_s_semant, X_s_semant, Y_s_semant = obtain_data_for_feature(gen_folder+"S_Semantic/", "R.S.Semantic Feature")

assert np.array_equal(X_phase, X_phrase)
assert np.array_equal(X_phrase, X_g_semant)
assert np.array_equal(X_s_semant, X_g_semant)

assert np.array_equal(A_phase, A_phrase)
assert np.array_equal(A_phrase, A_g_semant)
assert np.array_equal(A_s_semant, A_g_semant)

assert np.array_equal(Y_phase[:,:2], Y_phrase[:,:2])
assert np.array_equal(Y_phrase[:,:2], Y_g_semant[:,:2])
assert np.array_equal(Y_g_semant[:,:2], Y_s_semant[:,:2])

# Text and Audio are always the same
X_all = X_phrase
A_all = A_phrase
print("\nTotal Text shape: ", X_all.shape)
print("Total Audio shape: ", A_all.shape)

# Combine all the features together
Y_all = np.concatenate((Y_s_semant, Y_g_semant[:,2:], Y_phrase[:,2:],  Y_phase[:,2:]), axis=1)
print("Total features shape: ", Y_all.shape)


### Save new files

new_Y_file_name = gen_folder + "AllTogether/train_n_val_Y_all.npy"
np.save(new_Y_file_name, Y_all)

new_X_file_name = gen_folder + "AllTogether/train_n_val_X_all.npy"
np.save(new_X_file_name, X_all)

new_A_file_name = gen_folder + "AllTogether/train_n_val_A_all.npy"
np.save(new_A_file_name, A_all)