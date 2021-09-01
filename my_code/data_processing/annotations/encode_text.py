from tqdm import tqdm
from os.path import join
from transformers import DistilBertTokenizer, DistilBertModel
import fasttext
import fasttext.util
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


def BERT_text_to_feat(tokenizer, model, text):
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
    sub_word_encodings = last_hidden_states[0, :].detach()
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

        merged_word_enc = torch.stack(curr_enc, dim=0).mean(dim=0)
        word_encodings.append(merged_word_enc)
        token_idx += 1

    text_enc = torch.stack(word_encodings)

    assert len(text_enc) == len(text)

    return text_enc


def FastText_text_to_feat(model, text):
    """
    Encode given text into FastText feature vector
    Args:
        model:       FastText model itself
        text:        text to be encoded

    Returns:
        text_enc     resulting feature vector

    """

    # Encode the whole text
    text_enc = [model[clean_str(str(word))] for word in text]

    assert len(text_enc) == len(text)

    return torch.Tensor(text_enc)


def encode_text(elan_object, hdf5_dataset, language_model, tokenizer=None, model=None ):
    """
    Encode features of a current file and save into hdf5 dataset
    Args:
        elan_object:          ELAN object containing the annotations
        hdf5_dataset:         the hdf5 dataset where the vectors will be saved
        language_model:       NLP model to use, such as BERT or FastText
        tokenizer:            BERT tokenizer
        model:                BERT model itself

    Returns:
        nothing, but the text features are saved into 'hdf5_dataset'
    """
    text_tier = elan_object.tiers["R.S.Form"][0]
    timeslot = elan_object.timeslots

    # Extract text to encode it first
    text = []

    for value in text_tier.values():
        st_t, end_t, word, _ = value

        assert word is not None

        word = word.lstrip()
        if word != "":
            text.append(word)

    # Split text into short enough parts
    numb_parts = len(text) // 400 + 1
    
    full_text_enc = []
    for part in range(numb_parts):
        text_part = text[part*400:(part+1)*400]

        # Encode the current part
        if language_model == "BERT":
            text_part_enc = BERT_text_to_feat(tokenizer, model, text_part)
        elif language_model == "FastText":
            text_part_enc = FastText_text_to_feat(model, text_part)
        else:
            raise NotImplementedError("The language model ", language_model, " is not implemented yet!")

        full_text_enc.append(text_part_enc)

    # Concatenate the text parts into a single tensor
    full_text_enc = torch.cat(full_text_enc, 0)

    # Now encode all the words together with their timing information
    curr_column_features = []

    word_id = 0
    for value in text_tier.values():
        st_t, end_t, word, _ = value

        if word is not None and word.lstrip() != "":
            time_n_feat = [timeslot[st_t] / 1000] + [timeslot[end_t] / 1000] + list(full_text_enc[word_id])

            curr_column_features.append(np.array(time_n_feat))

            word_id += 1

    curr_column_features = np.array(curr_column_features)

    hdf5_dataset.create_dataset(name="text", data=curr_column_features.astype(np.float64))


def create_hdf5_file(annotation_filename, target_dir):
    """
    Create the output hdf5 object based on the ELAN filename.
    """
    file_idx = annotation_filename[:2]
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    hdf5_file_name = join(target_dir, f"{file_idx}_text.hdf5")
    
    assert os.path.isfile(hdf5_file_name) == False
    
    return h5py.File(name=hdf5_file_name, mode='w')

if __name__ == "__main__":

    annotation_folder = "/home/tarask/Documents/Datasets/SaGa/Processed/Aug/transcripts/"

    text_folder ="/home/tarask/Documents/Datasets/SaGa/Processed/Aug/processed/word_vectors/train_n_val/"

    NLP_model = "FastText"
    word_embedding_dim = 300

    if NLP_model == "BERT":
        # create DistilBERT model
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-german-cased')
        model = DistilBertModel.from_pretrained('distilbert-base-german-cased')
    elif NLP_model == "FastText":
        print("Loading FastText model ...")
        # Load FastText model
        tokenizer = None
        fasttext.util.download_model('de', if_exists='ignore')  # German
        model = fasttext.load_model(f"cc.de.{word_embedding_dim}.bin")
    else:
        raise NotImplementedError("The language model " + NLP_model + " is not implemented yet!")

    # go though the gesture features
    for filename in tqdm(os.listdir(annotation_folder)):
        if not filename.endswith("eaf"):
            continue
        
        annotation_file = join(annotation_folder, filename)
        elan_object = pympi.Elan.Eaf(file_path=annotation_file)
        
        hdf5_dataset = create_hdf5_file(filename, text_folder)
        
        encode_text(elan_object, hdf5_dataset, NLP_model, tokenizer, model )

        hdf5_dataset.close()