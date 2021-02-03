import pympi
import numpy as np
import re
import os
import pickle
import h5py


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
            else:
                break

            for key, (st_t, end_t, label, _) in curr_tier.items():
                if label is not None:
                    # remove leading whitespace
                    label = label.lstrip()
                    if label != "" and label != " ":
                        if label.find('-')!=-1:
                            split = re.split('-',label)
                            for subval in split:
                                curr_values.append(subval.strip())
                        elif label.find('/')!=-1:
                            split = re.split('/',label)
                            for subval in split:
                                curr_values.append(subval.strip())
                        elif label.find(',')!=-1:
                            split = re.split(',',label)
                            for subval in split:
                                curr_values.append(subval.strip())
                        elif label.find('\n')!=-1:
                            split = re.split('\n',label)
                            for subval in split:
                                curr_values.append(subval.strip())
                                break
                        elif label == "relative Positionm Amount":
                            curr_values.append("relative Position")
                            curr_values.append("Amount")
                        elif label == "relatie Position":
                            curr_values.append("relative Position")
                        elif label == "Shape38":
                            curr_values.append("Shape")
                        else:
                            curr_values.append(label)

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

    print("Done!")


def encode_features(dict_file, curr_file, columns_to_consider):
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
    hf = h5py.File(name = hdf5_file_name, mode = 'a')

    for column in columns_to_consider:

        curr_dict = total_dict[column]

        curr_column_features = []

        if column in curr_tiers:
            curr_tier = curr_tiers[column][0]
        else:
            break

        for key, (st_t, end_t, label, _) in curr_tier.items():

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

                    #print(label)
                    for key in curr_dict:
                        if label.find(curr_dict[key])!=-1:
                            features[key] = 1
                            #print(key, 'corresponds to', curr_dict[key])

                    #print(time_key[st_t] / 1000, time_key[end_t] / 1000)

                    #print(features)

                    #print(" .... \n")

                    time_n_feat = [time_key[st_t] / 1000] + [time_key[end_t] / 1000] + features

                    curr_column_features.append(np.array(time_n_feat))

        print(column)

        curr_column_features = np.array(curr_column_features)

        hf.create_dataset(column, data=curr_column_features)


    hf.close()


if __name__ == "__main__":

    curr_folder = "/home/tarask/Documents/Datasets/SaGa/All_the_transcripts/"

    columns_to_consider = ["R.G.Left.Phrase", "R.G.Left.Phase", "R.G.Right.Phrase", "R.G.Right.Phase",
                 "R.Movement_relative_to_other_Hand", "R.S.Pos",
                 "R.G.Right Semantic", "R.S.Semantic Feature"]

    dict_file = "dict.pkl"

    #create_dict(curr_folder, columns_to_consider, dict_file)

    # go though the gesture features
    for item in os.listdir(curr_folder):
        if item[-3:] != "eaf":
            continue
        curr_file = curr_folder + item

        encode_features(dict_file, curr_file, columns_to_consider)