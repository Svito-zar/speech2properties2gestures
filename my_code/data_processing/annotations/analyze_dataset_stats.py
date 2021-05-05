import numpy as np


gen_folder = "/home/tarask/Documents/Datasets/SaGa/Processed/feat/"
gen_folder = "/home/tarask/Documents/Datasets/SaGa/Processed/feat/TextBased/AllTogether/"

file_name = gen_folder + "train_n_val_Y_all.npy"

X = np.load(file_name, allow_pickle=True)

print("Dataset shape: ", X.shape)

for data_num in range(2, 26):
    if data_num == 13 and data_num == 17:
        continue
    curr_file = X[X[:,0] == data_num]
    print("\ndata_num: ", data_num, " - ", curr_file.shape)

    # count "has Phase"
    for feat in range(6, 10):
        sum = np.sum(curr_file[:, 2+ feat])
        print("Feat ", feat - 6, " is present in ", sum, " frames")

exit(0)

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


# Create Text2Binary dataset
t2b = X[:, :3]
for ind, feat in enumerate(X):
    t2b[ind, 2] = np.max(feat[-9:-5])

print(np.sum(t2b[:,2:]))
print(t2b.shape)

exit(0)

for feat in range(19):
    column = X[:, 2 + feat]
    print(feat, np.sum(column))

print(X.shape)