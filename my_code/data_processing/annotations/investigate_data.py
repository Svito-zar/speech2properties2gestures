import pympi
import numpy as np
import re
import os

curr_folder = "/home/tarask/Documents/Datasets/SaGa/All_the_transcripts/"
curr_file = curr_folder + "/01_video_NEW.eaf"
elan = pympi.Elan.Eaf(file_path=curr_file)

# Detail

gen_tiers = ["R.G.Right.Practice", "R.G.Left.Practice", "R.G.Left.Phrase", "R.G.Right.Phrase",
             "R.G.Left.Phase", "R.G.Right.Phase",
              "R.Movement_relative_to_other_Hand", "R.S.Pos",
              "R.G.Right Semantic", "R.G.Right Semantic", "R.S.Semantic Feature"]


def fix_typos(label):
    if label == "relatie Position":
        label_fixed ="relative Position"
    elif label == "Shape38":
        label_fixed = "Shape"
    elif label == "Entities" or label == "Enitity":
        label_fixed = "Entity"
    else:
        label_fixed = label
    return label_fixed

def clean_label(label):
    # remove leading whitespace
    label = label.lstrip()

    if label.find('-') != -1:
        split = re.split('-', label)
        label_cleaned = [fix_typos(subval.strip()) for subval in split]
    elif label.find('/') != -1:
        split = re.split('/', label)
        label_cleaned = [fix_typos(subval.strip()) for subval in split]
    elif label.find(',') != -1:
        split = re.split(',', label)
        label_cleaned = [fix_typos(subval.strip()) for subval in split]
    elif label.find('\n') != -1:
        split = re.split('\n', label)
        label_cleaned = [split[0].strip()]
    elif label == "relative Positionm Amount":
        label_cleaned = ["relative Position","Amount"]
    else:
        label_cleaned = [fix_typos(label.strip())]

    return label_cleaned


if __name__ == "__main__":

    # go though the gesture features
    for column in gen_tiers:

        curr_values = []

        # go through all the files in the dataset

        print(column)

        for item in os.listdir(curr_folder):
            if item[-3:] != "eaf":
                continue
            curr_file = curr_folder + item

            elan = pympi.Elan.Eaf(file_path=curr_file)
            curr_tiers = elan.tiers

            if column in curr_tiers:
                #print(item)
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
                elif len(value) == 3:
                    (st_t, label, _) = value
                elif len(value) == 2:
                    (st_t, label) = value
                if label is not None and label != "" and label != " ":
                    label_cleaned = clean_label(label)
                    for label_parts in label_cleaned:
                            curr_values.append(label_parts)

        if len(set(curr_values)) > 0:
            print(column, len(set(curr_values)))
            unique_values = np.array(set(curr_values))
            print(unique_values)

            print("\n")

        else:
            print(column, " is empty ...")
            print(curr_values)

    print("Done!")