import pympi
import numpy as np
import re
import os

curr_folder = "/home/tarask/Documents/Datasets/SaGa/All_the_transcripts/"
curr_file = curr_folder + "/01_video_NEW.eaf"
elan = pympi.Elan.Eaf(file_path=curr_file)


gen_tiers = [ "R.G.Left.Phrase", "R.G.Left.Phase", "R.G.Right.Phrase", "R.G.Right.Phase",
              "R.Movement_relative_to_other_Hand", "R.S.Form", "R.S.Pos",
              "R.G.Right Semantic", "R.S.Semantic Feature"]


#time_key = elan.timeslots

# go though the gesture features
for column in gen_tiers:

    curr_values = []

    # go through all the files in the dataset

    for item in os.listdir(curr_folder):
        if item[-3:] != "eaf":
            continue
        curr_file = curr_folder + item

        elan = pympi.Elan.Eaf(file_path=curr_file)
        curr_tiers = elan.tiers

        if column in curr_tiers:
            #print(item)
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

    if len(set(curr_values)) > 0:
        print(column, len(set(curr_values)))
        unique_values = np.array(set(curr_values))
        print(unique_values)

        print("\n")

print("Done!")