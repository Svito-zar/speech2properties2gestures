from transformers import DistilBertTokenizer, DistilBertModel
import torch
import pympi
import numpy as np
import os
import h5py


def text_to_feat(tokenizer, model, text):
    """
    Encode given text into BERT feature vector
    Args:
        tokenizer:   BERT tokenizer
        model:       BERT model itself
        text:        text to be encoded

    Returns:
        text_enc     resulting feature vector

    """
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    text_enc = last_hidden_states[0, 1:-1]

    return text_enc

def encode_text(elan_file, hdf5_file_name):
    """
    Encode features of a current file and save into hdf5 dataset
    Args:
        elan_file:            file with the ELAN annotations
        hdf5_file_name:       file for storing the pre=processed features

    Returns:
        nothing, saves a new hdf5 file
    """

    elan = pympi.Elan.Eaf(file_path=elan_file)
    curr_tier = elan.tiers["R.S.Form"][0]
    time_key = elan.timeslots

    # create hdf5 file
    assert os.path.isfile(hdf5_file_name) == False
    hf = h5py.File(name=hdf5_file_name, mode='a')

    # Extract text to encode it first
    text = []

    for key, value in curr_tier.items():
        (st_t, end_t, word, _) = value

        if word is not None:
            word = word.lstrip()
            if word != "" and word != " ":
                text.append(word)

    # create DistilBERT model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-german-cased')
    model = DistilBertModel.from_pretrained('distilbert-base-german-cased')

    # Split text into short enough parts
    numb_parts = len(text) // 500 + 1
    full_text_enc = []
    for part in range(numb_parts):
        text_part= text[part*500:(part+1)*500]

        # Encode the current part
        text_part_enc = text_to_feat(tokenizer, model, text_part)

        full_text_enc.append(text_part_enc)

    # Combine two halfs together
    full_text_enc = torch.cat(full_text_enc, 0)
    print(full_text_enc.shape)

    # Now encode all the words together with their timing information
    curr_column_features = []

    word_id = 0
    for key, value in curr_tier.items():
        (st_t, end_t, word, _) = value

        if word is not None and word != "" and word != " ":

            time_n_feat = [time_key[st_t] / 1000] + [time_key[end_t] / 1000] + list(full_text_enc[word_id])

            curr_column_features.append(np.array(time_n_feat))

            word_id+=1

    curr_column_features = np.array(curr_column_features)

    print(curr_column_features.shape)

    hf.create_dataset(name="text", data=curr_column_features.astype(np.float64))

    hf.close()

if __name__ == "__main__":

    curr_folder = "/home/tarask/Documents/Datasets/SaGa/All_the_transcripts/"

    # go though the gesture features
    for item in os.listdir(curr_folder):
        if item[-3:] != "eaf":
            continue
        elan_file = curr_folder + item
        feature_file = "feat/" + item[:-3] + "-text.hdf5"

        encode_text(elan_file, feature_file)