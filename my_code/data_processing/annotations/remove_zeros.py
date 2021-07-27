from os.path import join
import numpy as np


dataset_dir = "../../../dataset/processed/numpy_arrays/train_n_val/"

load_dataset = lambda fname : np.load(join(dataset_dir, fname), allow_pickle=True)
save_dataset = lambda fname, arr : np.save(join(dataset_dir, fname), arr)

full_audio_dataset = load_dataset("Audio.npy")
full_text_dataset = load_dataset("Text.npy")
print("Audio:", full_audio_dataset.shape)
print("Text:", full_text_dataset.shape)

for property_name in ["Phase", "Semantic", "Phrase"]:
    property_dataset = load_dataset(f"{property_name}_properties.npy")
    print(f"{property_name}:", property_dataset.shape)    
    feat_sum = np.sum(property_dataset[:, 2:], axis=1)
    zero_ids = np.where(feat_sum == 0)

    property_dataset     = np.delete(property_dataset, zero_ids, axis=0)
    curr_audio_dataset   = np.delete(full_audio_dataset, zero_ids, axis=0)
    curr_text_dataset    = np.delete(full_text_dataset, zero_ids, axis=0)
    print("--->", property_dataset.shape)    
    
    save_dataset(f"{property_name}_nozeros_properties", property_dataset)
    save_dataset(f"{property_name}_nozeros_audio", curr_audio_dataset)
    save_dataset(f"{property_name}_nozeros_text", curr_text_dataset)
