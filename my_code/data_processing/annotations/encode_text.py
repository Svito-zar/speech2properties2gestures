from transformers import DistilBertTokenizer, DistilBertModel
import torch
import pympi
import numpy as np
import os
import h5py


def clean_str(word):
    """
    Data cleaning for the text
    Args:
        word:         input word string

    Returns:
        final_word:   cleaned version of the word string

    """

    new_word = word.replace(" ", "").replace("s...", "so").replace("???", "").replace("-", "_")

    special_chars = ".!@#$%^&*_/><"

    for sp_char in special_chars:
        if sp_char in new_word:
            new_word = new_word.replace(sp_char, "")

    final_word = new_word.replace(" ", "")

    if final_word == "":
        final_word = "ah"

    return final_word


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

    # Clean text first
    text_str = " ".join(clean_str(str(x)) for x in text)

    # Encode the whole text split into sub-words
    input_ids = torch.tensor(tokenizer.encode(text_str, add_special_tokens=False)).unsqueeze(0)
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    sub_word_encodings = last_hidden_states[0, :]

    sub_word_tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist()[0])

    # For each word, we merge the sub-word embeddings (if there are more than one) into a single vector 
    word_encodings = []
    token_idx = 0
    while token_idx < len(sub_word_tokens):
        curr_enc = [sub_word_encodings[token_idx]]
        if token_idx < len(sub_word_tokens) - 1:
            # Related sub-words are denoted by a # character
            # e.g. the word 'incoming' may be split into [in, #com, #ing]
            while sub_word_tokens[token_idx + 1][0] == "#":
                token_idx += 1
                curr_enc.append(sub_word_encodings[token_idx])
                if token_idx == len(sub_word_tokens) - 1:
                    break
        merged_word_enc = np.mean(curr_enc, keepdims=1)
        merged_word_enc = merged_word_enc[0]
        word_encodings.append(merged_word_enc)
        token_idx += 1

    text_enc = torch.stack(word_encodings)

    assert len(text_enc) == len(text)

    return text_enc


def encode_text(tokenizer, model,elan_file, hdf5_file_name):
    """
    Encode features of a current file and save into hdf5 dataset
    Args:
        tokenizer:            BERT tokenizer
        model:                BERT model itself
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
            if word != "":
                text.append(word)

    # Split text into short enough parts
    numb_parts = len(text) // 400 + 1
    
    full_text_enc = []
    for part in range(numb_parts):
        text_part = text[part*400:(part+1)*400]

        # Encode the current part
        text_part_enc = text_to_feat(tokenizer, model, text_part)

        full_text_enc.append(text_part_enc)

    # Concatenate the text parts into a single tensor
    full_text_enc = torch.cat(full_text_enc, 0)

    # Now encode all the words together with their timing information
    curr_column_features = []

    word_id = 0
    for key, value in curr_tier.items():
        (st_t, end_t, word, _) = value

        if word is not None and word.lstrip() != "":
            time_n_feat = [time_key[st_t] / 1000] + [time_key[end_t] / 1000] + list(full_text_enc[word_id])

            curr_column_features.append(np.array(time_n_feat))

            word_id += 1

    curr_column_features = np.array(curr_column_features)

    hf.create_dataset(name="text", data=curr_column_features.astype(np.float64))

    hf.close()

if __name__ == "__main__":

    # create DistilBERT model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-german-cased')
    model = DistilBertModel.from_pretrained('distilbert-base-german-cased')

    curr_folder = "/home/tarask/Documents/Datasets/SaGa/Raw/All_the_transcripts/"

    # go though the gesture features
    for item in os.listdir(curr_folder):
        if item[-3:] != "eaf":
            continue
        elan_file = curr_folder + item
        print(elan_file)

        feature_file = "feat/" + item[:2] + "_text.hdf5"

        encode_text(tokenizer, model, elan_file, feature_file)