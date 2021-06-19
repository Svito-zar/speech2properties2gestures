from tqdm import tqdm
from os import path
from os.path import join
import pympi
import numpy as np
import os
import pickle
import h5py
from pprint import pprint
from data_processing.annotations import utils

def open_elan_tier_for_property(elan_file, property_name):
    if property_name not in elan_file.tiers:
        raise KeyError()
    
    tier = elan_file.tiers[property_name]

    return tier[0] if len(tier[0]) > 0 else tier[1]

def open_or_create_label_dict(annotation_folder, dict_file):
    """
    Open or create a dictionary that maps property names to their index-label pairs.

    Args:
        annotation_folder:  path to the annotation folder
        dict_file:  the save path of the dictionary

    Returns:
        The dictionary that maps properties to their labels.
    """
    if path.isfile(dict_file):
        with open(dict_file, 'rb') as handle:
            print(f"Opening label dictionary: '{dict_file}'.")
            return pickle.load(handle)

    print(f"Creating label dictionary: '{dict_file}'.")
    label_dict = {}

    progress_bar = tqdm(ALL_PROPERTIES)
    for property_name in progress_bar:
        progress_bar.set_description("Parsing property")
        possible_labels = set()

        # go through all the files in the dataset
        for filename in os.listdir(annotation_folder):
            if not filename.endswith("eaf"):
                continue

            annotation_path = join(annotation_folder, filename)
            elan_file = pympi.Elan.Eaf(file_path=annotation_path)

            try:
                elan_tier = open_elan_tier_for_property(elan_file, property_name)
            except KeyError:
                # TODO(RN) This used to be break which I think is incorrect
                continue

            for annotation_entry in elan_tier.values():
                assert len(annotation_entry) == 4
                (st_t, end_t, label, _) = annotation_entry
                # Sometimes they are messed up
                if label is None:
                    (st_t, label, _, _) = annotation_entry

                if label is not None and label != "" and label != " ":
                    cleaned_label = utils.clean_and_split_label(label)
                    for label_parts in cleaned_label:
                        possible_labels.add(label_parts)
        
        # Explicitly store the label indices
        label_dict[property_name] = {i: val for i, val in enumerate(possible_labels)}

    # Save the dictionary
    f = open(dict_file, "wb")
    pickle.dump(label_dict, f)
    f.close()

    return label_dict

def encode_other_features(total_dict, curr_file, columns_to_consider):
    """
    Encode features of a current file and save into hdf5 dataset
    Args:
        total_dict:           dictionary for the binary coding
        curr_file:            file with the ELAN annotations
        columns_to_consider:  which columns are we interested in

    Returns:
        nothing, saves a new hdf5 file
    """

    elan = pympi.Elan.Eaf(file_path=curr_file)
    curr_tiers = elan.tiers
    time_key = elan.timeslots

    # create hdf5 file
    hdf5_file_name = "feat/" + curr_file[61:63] + "_feat.hdf5"
    assert os.path.isfile(hdf5_file_name) == False
    hf = h5py.File(name=hdf5_file_name, mode='a')

    for column in columns_to_consider:

        curr_dict = total_dict[column]

        curr_column_features = []

        if column in curr_tiers:
            curr_tier = curr_tiers[column][0]
            if len(curr_tier) == 0:
                curr_tier = curr_tiers[column][1]
        else:
            continue

        for key, value in curr_tier.items():
            if len(value) == 4:
                (st_t, end_t, label, _) = value
                # sometime they are messed up
                if label is None and _ is None:
                    (st_t, label, _, _) = value

            if label is not None:
                # remove leading whitespace
                label = label.lstrip()
                if label != "" and label != " ":
                    if label == "relative Positionm Amount":
                        label = "relative Position, Amount"
                    elif label == "relatie Position":
                        label = "relative Position"
                    elif label == "Shape38":
                        label = "Shape"

                    features = [0 for _ in range(len(curr_dict))]

                    for key in curr_dict:
                        if label.find(curr_dict[key]) != -1:
                            features[key] = 1

                    if time_key[st_t] is None or time_key[end_t] is None:
                        continue

                    time_n_feat = [time_key[st_t] / 1000] + [time_key[end_t] / 1000] + features

                    curr_column_features.append(np.array(time_n_feat))

        curr_column_features = np.array(curr_column_features)

        hf.create_dataset(column, data=curr_column_features)

    hf.close()


def encode_main_g_features(total_dict, curr_file):
    """
       Encode main gesture features of a current file and save into hdf5 dataset
       Args:
           total_dict:           dictionary for the binary coding
           curr_file:            file with the ELAN annotations
       Returns:
           nothing, saves features in hdf5 file
       """


    elan = pympi.Elan.Eaf(file_path=curr_file)
    curr_tiers = elan.tiers
    time_key = elan.timeslots

    # create hdf5 file
    hdf5_file_name = "feat/" + curr_file[61:63] + "_feat.hdf5"
    hf = h5py.File(name=hdf5_file_name, mode='a')

    hands = ["Right", "Left"]
    #main_columns = ["R.G.Right.Practice", "R.G.Left.Practice", "R.G.Left.Phrase", "R.G.Right.Phrase"]

    for hand in hands:

        phrase = "R.G." + hand + ".Phrase"
        practice = "R.G." + hand + ".Practice"

        curr_phrase_dict = total_dict[phrase]
        curr_practice_dict = total_dict[practice]

        curr_column_features = []

        if phrase in curr_tiers:
            curr_phrase_tier = curr_tiers[phrase][0]
            if len(curr_phrase_tier) == 0:
                curr_phrase_tier = curr_tiers[phrase][1]
        else:
            break

        if practice in curr_tiers:
            curr_practice_tier = curr_tiers[practice][0]
            if len(curr_practice_tier) == 0:
                curr_practice_tier = curr_tiers[practice][1]
        else:
            break

        for key, value in curr_practice_tier.items():
            (ges_key, practice_val, _, _) = value

            if practice_val is not None:
                # remove leading whitespace
                practice_val = practice_val.lstrip()
                if practice_val != "" and practice_val != " ":

                    pract_features = [0 for _ in range(len(curr_practice_dict))]

                    for dict_key in curr_practice_dict:
                        if practice_val.find(curr_practice_dict[dict_key]) != -1:
                            pract_features[dict_key] = 1

                    (st_t, end_t, phrase_val, _)  = curr_phrase_tier[ges_key]

                    if time_key[st_t] is None or time_key[end_t] is None:
                        continue

                    if phrase_val is not None:
                        # remove leading whitespace
                        phrase_val = phrase_val.lstrip()
                        if phrase_val != "" and phrase_val != " ":

                            phr_features = [0 for _ in range(len(curr_phrase_dict))]

                            for dict_key in curr_phrase_dict:
                                if phrase_val.find(curr_phrase_dict[dict_key]) != -1:
                                    phr_features[dict_key] = 1

                    time_n_feat = [time_key[st_t] / 1000] + [time_key[end_t] / 1000] + phr_features + pract_features

                    curr_column_features.append(np.array(time_n_feat))

        curr_column_features = np.array(curr_column_features)

        hf.create_dataset("gesture_phrase_n_practice_"+hand, data=curr_column_features)

    hf.close()


def encode_g_semant(total_dict, curr_file):
    """
       Encode gesture semantic features of a current file and save into hdf5 dataset
       Args:
           total_dict:           dictionary for the binary coding
           curr_file:            file with the ELAN annotations
       Returns:
           nothing, saves features in hdf5 file
       """


    elan = pympi.Elan.Eaf(file_path=curr_file)
    curr_tiers = elan.tiers
    time_key = elan.timeslots

    # create hdf5 file
    hdf5_file_name = "feat/" + curr_file[61:63] + "_feat.hdf5"
    hf = h5py.File(name=hdf5_file_name, mode='a')

    hands = ["Right", "Left"]
    #main_columns = ["R.G.Left.Phrase", "R.G.Right.Phrase", "R.G.Right Semantic", "R.G.Left Semantic"]

    for hand in hands:

        phrase = "R.G." + hand + ".Phrase"
        semant = "R.G." + hand + " Semantic"

        # Take dictionary mapping labels into numbers
        curr_semant_dict = total_dict[semant]

        # empty list to store the features
        curr_column_features = []

        # read phrase tier
        if phrase in curr_tiers:
            curr_phrase_tier = curr_tiers[phrase][0]
            if len(curr_phrase_tier) == 0:
                curr_phrase_tier = curr_tiers[phrase][1]
        else:
            print("WARNING: A file " + curr_file + " is ignored for " + hand + " hand since it contains no " + hand + " PHRASE tier")
            continue

        # read semant tier
        if semant in curr_tiers:
            curr_semant_tier = curr_tiers[semant][0]
            if len(curr_semant_tier) == 0:
                curr_semant_tier = curr_tiers[semant][1]
        else:
            print("WARNING: A file " + curr_file + " is ignored for " + hand + " hand since it contains no " + hand + " SEMANTIC tier")
            continue

        # go over all the semantic labels
        for key, value in curr_semant_tier.items():
            (sem_st_t, sem_end_t, semant_val, _) = value

            if semant_val is not None:
                # remove leading whitespace
                semant_val = semant_val.lstrip()
                if semant_val != "" and semant_val != " ":

                    semant_features = [0 for _ in range(len(curr_semant_dict))]

                    # map a string (semant_val) to a binary vector (semant_features)
                    for dict_key in curr_semant_dict:
                        if semant_val.find(curr_semant_dict[dict_key]) != -1:
                            semant_features[dict_key] = 1

                    # find the Phrase it corresponds to
                    for key, value in curr_phrase_tier.items():
                        (phr_st_t, phr_end_t, phrase_val, _) = value
                        if time_key[phr_st_t] is None or time_key[phr_end_t] is None:
                            continue
                        if time_key[phr_end_t] >= time_key[sem_st_t]:
                            break

                    # use timing for the whole Phrase
                    time_n_feat = [time_key[phr_st_t] / 1000] + [time_key[phr_end_t] / 1000] + semant_features

                    curr_column_features.append(np.array(time_n_feat))

        curr_column_features = np.array(curr_column_features)

        hf.create_dataset(semant, data=curr_column_features)

    hf.close()


if __name__ == "__main__":
    ALL_PROPERTIES = [
        'R.G.Left Semantic', 'R.G.Right Semantic',
        'R.G.Left.Phase',    'R.G.Right.Phase',
        'R.G.Left.Phrase',   'R.G.Right.Phrase',
        'R.G.Left.Practice', 'R.G.Right.Practice',
        'R.Movement_relative_to_other_Hand',
        'R.S.Pos' ,
        'R.S.Semantic Feature'
    ]

    TOP_LEVEL_PROPERTIES = [
        "R.G.Left.Phase", "R.G.Right.Phase",
        "R.G.Left.Phrase", "R.G.Right.Phrase",
        "R.S.Semantic Feature"
    ]
    
    annotation_folder = "/home/work/Desktop/repositories/probabilistic-gesticulator/dataset/All_the_transcripts/"

    # create_dict(annotation_folder, "dict.pkl")

    dict_file = "dict_new.pkl"
    label_dict = open_or_create_label_dict(annotation_folder, dict_file)
    pprint(label_dict)

    with open(dict_file, 'rb') as handle:
        total_dict = pickle.load(handle)

    print(total_dict)

    columns_to_consider = ["R.G.Left.Phase", "R.G.Right.Phase",
                           "R.G.Left.Phrase", "R.G.Right.Phrase",  "R.S.Semantic Feature"]
                           #"R.G.Right Semantic", "R.G.Left Semantic",]

    # go though the gesture features
    for filename in sorted(os.listdir(annotation_folder)):
        if not filename.endswith(".eaf"):
            continue

        annotation_file = join(annotation_folder, filename)

        print(path.basename(annotation_file))

        encode_other_features(label_dict, annotation_file, TOP_LEVEL_PROPERTIES)

        encode_main_g_features(label_dict, annotation_file)

        encode_g_semant(label_dict, annotation_file)

        # file_idx = path.basename(annotation_file)[:2]
        # feature_file = join("feat/", f"{file_idx}_feat.hdf5")
        # hf = h5py.File(name=feature_file, mode='r')
        # # print(len(hf.keys()), hf.keys())
