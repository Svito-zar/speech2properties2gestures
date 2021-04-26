import numpy as np

# Get the dataset with all the labels
gen_folder = "/home/tarask/Documents/Datasets/SaGa/Processed/feat/TextBased/AllTogether/"
file_name = gen_folder + "train_n_val_Y_all.npy"
X = np.load(file_name, allow_pickle=True)

print("Dataset shape: ", X.shape)

# count "no feat"
no_f = 0
for feat in X:
    if np.sum(feat[2:]) == 0:
        no_f += 1
print("No f: ", no_f)

# count "has Phase"
has_phase = 0
for feat in X:
    if np.sum(feat[-5:]) > 0:
        has_phase += 1
print("has phase", has_phase)

# count "has Phrase"
has_phase = 0
for feat in X:
    if np.sum(feat[-9:-5]) > 0:
        has_phase += 1
print("has phrase", has_phase)


# Create Text2Binary dataset (Y side of it)
t2b = X[:, :3]
for ind, feat in enumerate(X):
    t2b[ind, 2] = np.max(feat[-9:-5])

print(np.sum(t2b[:,2:]))
print(t2b.shape)


binary_gest_label_file = gen_folder + "train_n_val_Binary.npy"
np.save(binary_gest_label_file, t2b)