import pympi
import numpy as np
import re
import os

curr_folder = "/home/tarask/Documents/Datasets/SaGa/All_the_transcripts/"
curr_file = curr_folder + "/01_video_NEW.eaf"
elan = pympi.Elan.Eaf(file_path=curr_file)

gen_tiers = elan.tiers
#time_key = elan.timeslots

# go though the gesture features
for column in gen_tiers:

    curr_values_pre = []
    curr_values_post = []

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
                if label != "" and label != " ":
                    if False: #label.find('-')!=-1:
                        split = re.split('-',label)
                        curr_values_pre.append(pre)
                        curr_values_post.append(post)
                    else:
                        curr_values_pre.append(label)

    if len(set(curr_values_pre)) > 0:
        print(column, len(set(curr_values_pre)))
        unique_values = np.array(set(curr_values_pre))
        print(unique_values)

        print("\n")


print("Done!")