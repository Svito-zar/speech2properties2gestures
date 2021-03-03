import numpy as np


gen_folder = "/home/tarask/Documents/Datasets/SaGa/Processed/feat/S_Semantics/"
file_name = gen_folder + "train_Y_R.S.Semantic Feature.npy"

X = np.load(file_name, allow_pickle=True)

# count "no feat"
no_f = 0
for feat in X:
    if np.sum(feat[2:]) == 0:
        no_f += 1
print("No f: ", no_f)


for feat in range(8):
    column = X[:, 2 + feat]
    print(feat, np.sum(column))

print(X.shape)