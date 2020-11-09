import numpy as np

body = np.load("/home/taras/Documents/Datasets/From_Habibie/conan/test_conan.npz")['body']
print(body.shape)
# (87263, 64, 165)