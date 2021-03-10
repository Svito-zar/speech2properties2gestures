import numpy as np


gen_folder = "/home/tarask/Documents/Datasets/SaGa/Processed/feat/S_Semantics/"

# Merge train n val
prop = "R.S.Semantic Feature"

val_file_name = gen_folder + "val_Y_" + prop + ".npy"
Y_val = np.load(val_file_name, allow_pickle=True)
print(Y_val.shape)

train_file_name = gen_folder + "train_Y_" + prop + ".npy"
Y_train = np.load(train_file_name, allow_pickle=True)
print(Y_train.shape)

Y_train_n_val = np.concatenate((Y_train, Y_val), axis = 0)
print(Y_train_n_val.shape)


val_X_file_name = gen_folder + "val_X_" + prop + ".npy"
X_val = np.load(val_X_file_name, allow_pickle=True)
print(X_val.shape)

train_X_file_name = gen_folder + "train_X_" + prop + ".npy"
X_train = np.load(train_X_file_name, allow_pickle=True)
print(X_train.shape)

X_train_n_val = np.concatenate((X_train, X_val), axis = 0)
print(X_train_n_val.shape)



# Merge certain features together
direction = np.clip(Y_train_n_val[:, 3] + Y_train_n_val[:, 7], 0, 1)
direction = Y_train_n_val[:, 3] + Y_train_n_val[:, 7]
Y_train_n_val[:, 3] = direction
# remove "relative Position", which we already merged above
Y_train_n_val = np.delete(Y_train_n_val, 7, 1)
# remove "Deictic"
Y_train_n_val = np.delete(Y_train_n_val, 4, 1)
print(np.array(Y_train_n_val).shape)


### Save new files

new_Y_file_name = gen_folder + "train_n_val_Y_" + prop + ".npy"
np.save(new_Y_file_name, Y_train_n_val)

new_X_file_name = gen_folder + "train_n_val_X_" + prop + ".npy"
np.save(new_X_file_name, X_train_n_val)