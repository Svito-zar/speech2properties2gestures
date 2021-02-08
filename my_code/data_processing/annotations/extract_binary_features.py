import pympi
import numpy as np
import re
import os
import pickle
import h5py

from my_code.data_processing.annotations.investigate_data import clean_label


def create_dict(curr_folder, columns_to_consider, dict_file):
    """
    Build dictionary based on all the possible features in the dataset
    Args:
        curr_folder:         folder with the dataset
        columns_to_consider: columns to consider
        dict_file:           file to store the created dictionary (pickle)

    Returns:
        nothing

    """

    # Create dictionary for all the labels
    dict = {}

    # go though the gesture features
    for column in columns_to_consider:

        curr_values = []

        # go through all the files in the dataset
        for item in os.listdir(curr_folder):
            if item[-3:] != "eaf":
                continue
            curr_file = curr_folder + item

            elan = pympi.Elan.Eaf(file_path=curr_file)
            curr_tiers = elan.tiers

            if column in curr_tiers:
                curr_tier = curr_tiers[column][0]
                if len(curr_tier) == 0:
                    curr_tier = curr_tiers[column][1]
            else:
                break

            for key, value in curr_tier.items():
                if len(value) == 4:
                    (st_t, end_t, label, _) = value
                    # sometime they are messed up
                    if label is None and _ is None:
                        (st_t, label, _, _) = value
                if label is not None and label != "" and label != " ":
                    label_cleaned = clean_label(label)
                    for label_parts in label_cleaned:
                        curr_values.append(label_parts)

        # If this feature is actually present
        if len(set(curr_values)) > 0:

            # create a corresponding dictionary
            dict[column] = {}
            for idx, val in enumerate(set(curr_values)):
                dict[column][idx] = val

            print(column, len(set(curr_values)))
            unique_values = np.array(set(curr_values))
            print(unique_values)
            print("\n")

    f = open(dict_file, "wb")
    pickle.dump(dict, f)
    f.close()

    print(dict)

    print("Done!")


def encode_other_features(dict_file, curr_file, columns_to_consider):
    """
    Encode features of a current file and save into hdf5 dataset
    Args:
        dict_file:            file with a dictionary for the binary coding
        curr_file:            file with the ELAN annotations
        columns_to_consider:  which columns are we interested in

    Returns:
        nothing, saves a new hdf5 file
    """

    with open(dict_file, 'rb') as handle:
        total_dict = pickle.load(handle)

    elan = pympi.Elan.Eaf(file_path=curr_file)
    curr_tiers = elan.tiers
    time_key = elan.timeslots

    # create hdf5 file
    hdf5_file_name = "feat/" + curr_file[57:-4] + "_feat.hdf5"
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
            break

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


def encode_main_g_features(dict_file, curr_file):
    """
       Encode main gesture features of a current file and save into hdf5 dataset
       Args:
           dict_file:            file with a dictionary for the binary coding
           curr_file:            file with the ELAN annotations
       Returns:
           nothing, saves features in hdf5 file
       """

    with open(dict_file, 'rb') as handle:
        total_dict = pickle.load(handle)

    elan = pympi.Elan.Eaf(file_path=curr_file)
    curr_tiers = elan.tiers
    time_key = elan.timeslots

    # create hdf5 file
    hdf5_file_name = "feat/" + curr_file[57:-4] + "_feat.hdf5"
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

    return 0

if __name__ == "__main__":

    curr_folder = "/home/tarask/Documents/Datasets/SaGa/All_the_transcripts/"

    dict_file = "dict.pkl"

    # create_dict(curr_folder, columns_to_consider, dict_file)

    columns_to_consider = ["R.G.Left.Phase", "R.G.Right.Phase",
                           "R.Movement_relative_to_other_Hand", "R.S.Pos",
                           "R.G.Right Semantic", "R.G.Left Semantic", "R.S.Semantic Feature"]

    # go though the gesture features
    for item in os.listdir(curr_folder):
        if item[-3:] != "eaf":
            continue
        curr_file = curr_folder + item

        print(curr_file)

        encode_other_features(dict_file, curr_file, columns_to_consider)

        encode_main_g_features(dict_file, curr_file)