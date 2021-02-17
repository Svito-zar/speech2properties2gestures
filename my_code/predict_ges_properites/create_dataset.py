import h5py
import numpy as np
import os
import bisect


def create_dataset(general_folder, specific_subfolder, feature_name, dataset_name):
    """

    Args:
        general_folder:     folder where all the data is stored
        specific_subfolder: name of a specific subfolder
        feature_name:       name of the feature we are considering
        dataset_name:       name of the dataset

    Returns:
        Nothing, save dataset in npy file
    """

    curr_folder = general_folder + specific_subfolder + "/"

    # Initialize empty lists for the dataset input and output
    X_dataset = []
    Y_dataset = []

    # go though the dataset recordings
    for recording_id in range(25):
        feat_file = str(recording_id).zfill(2) + "_feat.hdf5"

        # if this recording belong to the current dataset
        if feat_file in os.listdir(curr_folder):

            print("Consider file number :", str(recording_id).zfill(2))

            feat_hf = h5py.File(name=curr_folder + feat_file, mode='r')
            spec_feat_hf = feat_hf.get(feature_name)

            if spec_feat_hf is None:
                print("Skip file with only the following keys:")
                print(len(feat_hf.keys()), feat_hf.keys())
                continue
            spec_feat = np.array(spec_feat_hf)

            text_file = str(recording_id).zfill(2) + "_text.hdf5"

            text_hf = h5py.File(name=curr_folder + text_file, mode='r')
            text_array = text_hf.get("text")
            text_timing = text_array[:, :2]

            for feat_id in range(spec_feat.shape[0]):

                curr_feat_vec = spec_feat[feat_id]

                # First two values contain st_time and end_time, other values - feature vector itself
                curr_feat_timing = curr_feat_vec[:2].round(1)
                curr_feat_values = curr_feat_vec[2:]

                # add some sample with no semantic gesture
                if feat_id > 0:

                    for time_st in np.arange(prev_feat_timing[1] + 0.5, prev_feat_timing[1] + 1.5, 0.4):

                        if time_st >= curr_feat_timing[0]:
                            break

                        # Save some extra info which might be useful later on
                        output_vector = np.concatenate(([recording_id, time_st.round(1)], [0 for _ in range(19)]))

                        # find the corresponding words
                        curr_word_id = bisect.bisect(text_timing[:, 0], time_st)

                        # encode current word with the next three and previous three words
                        # while also storing time offset from the current time-step
                        input_vector = [
                            np.concatenate(([text_timing[word_id, 0].round(1) - time_st.round(1)], text_array[word_id, 2:]))
                            for word_id in range(curr_word_id - 3, curr_word_id + 4)]

                        X_dataset.append(np.array(input_vector))
                        Y_dataset.append(output_vector)

                for time_st in np.arange(curr_feat_timing[0], curr_feat_timing[1], 0.4):

                    # Save some extra info which might be useful later on
                    output_vector = np.concatenate(([recording_id, time_st.round(1)], curr_feat_values))

                    # find the corresponding words
                    curr_word_id = bisect.bisect(text_timing[:, 0], time_st)

                    if curr_word_id + 4 > len(text_array):
                        continue

                    # encode current word with the next three and previous three words
                    # while also storing time offset from the current time-step
                    input_vector = [
                        np.concatenate(([text_timing[word_id, 0].round(1) - time_st.round(1)], text_array[word_id, 2:]))
                        for word_id in range(curr_word_id - 3, curr_word_id + 4)]

                    # upsample under-represented classes
                    if output_vector[20] == 1:
                        mulp_factor = 60
                    elif output_vector[5] == 1:
                        mulp_factor = 340
                    elif output_vector[16] == 1 or output_vector[12] == 1:
                        mulp_factor = 30
                    elif output_vector[13] == 1 or output_vector[14] == 1 or output_vector[17] == 1:
                        mulp_factor = 14
                    elif output_vector[7] == 1 or output_vector[11] == 1:
                        mulp_factor = 12
                    elif output_vector[15] == 1 or output_vector[3] == 1 or output_vector[10] == 1:
                        mulp_factor = 2
                    else:
                        mulp_factor = 1

                    for _ in range(mulp_factor):
                        X_dataset.append(np.array(input_vector))
                        Y_dataset.append(output_vector)

                prev_feat_timing = curr_feat_timing

            print(np.array(Y_dataset).shape)
            print(np.array(X_dataset).shape)

    # create dataset file
    Y = np.asarray(Y_dataset, dtype=np.float32)
    X = np.asarray(X_dataset, dtype=np.float32)
    np.save(gen_folder + dataset_name+ "_Y_" + feature_name + ".npy", Y)
    np.save(gen_folder + dataset_name + "_X_" + feature_name + ".npy", X)

if __name__ == "__main__":

    gen_folder = "/home/tarask/Documents/Datasets/SaGa/Processed/feat/"
    dataset_name = subfolder = "train"
    feature_name = "gesture_phrase_n_practice_Right"

    create_dataset(gen_folder, subfolder, feature_name, dataset_name)