import numpy as np


gen_folder = "/home/tarask/Documents/Datasets/SaGa/Processed/feat/EVERYTHING/"
gen_folder = "/home/tarask/Documents/Datasets/SaGa/Processed/feat/AudioBased/G_Semantic/"
#gen_folder = "/home/tarask/Documents/Datasets/SaGa/Processed/feat/S_Semantics/"
gen_folder = "/home/tarask/Documents/Datasets/SaGa/Processed/feat/Text_n_AudioBased/G_Semantic/"
gen_folder = "/home/tarask/Documents/Datasets/SaGa/Processed/feat/ProsBased/"
gen_folder = "/home/tarask/Documents/Datasets/SaGa/Processed/feat/Text_n_AudioBased/AllTogether/"
prop = "R.S.Semantic Feature"
prop = "Semantic"
prop = "Phase"
prop = "all"

file_name = gen_folder + "train_n_val_Y_" + prop + ".npy"
Y_train_n_val = np.load(file_name, allow_pickle=True)
print(Y_train_n_val.shape)

file_name = gen_folder + "train_n_val_X_" + prop + ".npy"
X_train_n_val = np.load(file_name, allow_pickle=True)
print(X_train_n_val.shape)

file_name = gen_folder + "train_n_val_A_" + prop + ".npy"
A_train_n_val = np.load(file_name, allow_pickle=True)
print(A_train_n_val.shape)


# identify "empty" vectors
feat_sum = np.sum(Y_train_n_val[:,2:], axis=1)
zero_ids = np.where(feat_sum == 0)

# remove "empty" vectors
Y_train_n_val_new = np.delete(Y_train_n_val, zero_ids, 0)
X_train_n_val_new = np.delete(X_train_n_val, zero_ids, 0)
A_train_n_val_new = np.delete(A_train_n_val, zero_ids, 0)

print(Y_train_n_val_new.shape)
print(X_train_n_val_new.shape)
print(A_train_n_val_new.shape)

n_features = 19
#n_features = 5
freq = np.zeros(n_features)
for feat in range(n_features):
        column = Y_train_n_val_new[:, 2 + feat]
        freq[feat] = np.sum(column) # These are the binary gesture properties

print(freq)

### Save new files

new_Y_file_name = gen_folder + "no_zero/train_n_val_Y_" + prop + ".npy"
np.save(new_Y_file_name, Y_train_n_val_new)

new_X_file_name = gen_folder + "no_zero/train_n_val_X_" + prop + ".npy"
np.save(new_X_file_name, X_train_n_val_new)

new_A_file_name = gen_folder + "no_zero/train_n_val_A_" + prop + ".npy"
np.save(new_A_file_name, A_train_n_val_new)