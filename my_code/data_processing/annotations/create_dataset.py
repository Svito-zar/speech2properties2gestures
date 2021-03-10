import h5py
import numpy as np
import os
import bisect


def create_dataset(general_folder, specific_subfolder, feature_name, dataset_name, feature_dim):
    """

    Args:
        general_folder:     folder where all the data is stored
        specific_subfolder: name of a specific subfolder
        feature_name:       name of the feature we are considering
        dataset_name:       name of the dataset
        feature_dim:        dimensionality of the feature considered

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
                Warning("Skip file with only the following keys:", len(feat_hf.keys()), feat_hf.keys())
                continue
            spec_feat = np.array(spec_feat_hf)

            text_file = str(recording_id).zfill(2) + "_text.hdf5"

            text_hf = h5py.File(name=curr_folder + text_file, mode='r')
            text_array = text_hf.get("text")
            text_timing = text_array[:, :2]

            # Consider all the time-frames except for the first and last three words, since we need three words for the contexts
            start_time = text_timing[3, 0].round(1)
            end_time = text_timing[-4, 0].round(1)
            duration = end_time - start_time
            total_number_of_frames = int(duration * 5) # 0.2s time-steps

            """ print("Start time: ", start_time)
            print("End time: ", end_time)
            print("Total numb of frames: ", total_number_of_frames) """

            # First save all the text features
            curr_file_X_data = np.zeros((total_number_of_frames, 7, 769))

            time_ind = 0
            for time_st in np.arange(start_time, end_time-0.1, 0.2):

                # find the corresponding words
                curr_word_id = bisect.bisect(text_timing[:, 0], time_st)

                # encode current word with the next three and previous three words
                # while also storing time offset from the current time-step
                input_vector = [
                    np.concatenate(([text_timing[word_id, 0].round(1) - time_st.round(1)], text_array[word_id, 2:]))
                    for word_id in range(curr_word_id - 3, curr_word_id + 4)]

                curr_file_X_data[time_ind] = np.array(input_vector)

                time_ind = int( (time_st.round(1) - start_time) * 5 )

            # Create dataset for Y features
            curr_file_Y_data = np.zeros((total_number_of_frames, feature_dim+2))

            for feat_id in range(spec_feat.shape[0]):

                curr_feat_vec = spec_feat[feat_id]

                # First two values contain st_time and end_time, other values - feature vector itself
                curr_feat_timing = curr_feat_vec[:2].round(1)
                curr_feat_values = curr_feat_vec[2:]

                for time_st in np.arange(curr_feat_timing[0], curr_feat_timing[1], 0.2):

                    # Save some extra info which might be useful later on
                    output_vector = np.concatenate(([recording_id, time_st.round(1)], curr_feat_values))

                    time_ind = int((time_st.round(1) - start_time) * 5)

                    curr_file_Y_data[time_ind] = output_vector

            if len(X_dataset) == 0:
                X_dataset = curr_file_X_data
                Y_dataset = curr_file_Y_data
            else:
                X_dataset = np.concatenate((X_dataset, curr_file_X_data))
                Y_dataset = np.concatenate((Y_dataset, curr_file_Y_data))

            print(np.asarray(X_dataset, dtype=np.float32).shape)
            print(np.asarray(Y_dataset, dtype=np.float32).shape)

    # create dataset file
    Y = np.asarray(Y_dataset, dtype=np.float32)
    X = np.asarray(X_dataset, dtype=np.float32)

    # upsample underrepresented classes
    print("Upsampling ...")
    X_ups, Y_ups = upsample(X, Y, feature_dim)

    # save files
    np.save(gen_folder + dataset_name+ "_Y_" + feature_name + ".npy", Y_ups)
    np.save(gen_folder + dataset_name + "_X_" + feature_name + ".npy", X_ups)


def upsample(X, Y, feature_length):
    """

    Args:
        X:                  input dataset
        Y:                  output dataset
        feature_length:     number of features in the dataset

    Returns:
        X_upsampled:        upsampled input dataset with equalized features frequencies
        Y_upsampled:        upsampled output dataset with equalized features frequencies

    """

    freq = np.zeros(feature_length)
    for feat in range(feature_length):
        column = Y[:, 2 + feat]
        freq[feat] = np.sum(column)

    max_freq = np.max(freq)
    multipliers = [int(2 * max_freq // freq[feat]) for feat in range(feature_length)]

    Y_upsampled = list(np.copy(Y))
    X_upsampled = list(np.copy(X))

    for ind in range(Y.shape[0]):
        multipl_factor = 1
        for feat in range(feature_length):
            if Y[ind, 2 + feat] == 1:
                multipl_factor = max(multipl_factor, multipliers[feat])
        if multipl_factor > 1:
            for _ in range(multipl_factor-1):
                Y_upsampled.append(Y[ind])
                X_upsampled.append(X[ind])

    X_upsampled = np.asarray(X_upsampled, dtype=np.float32)
    Y_upsampled = np.asarray(Y_upsampled, dtype=np.float32)

    freq = np.zeros(feature_length)
    for feat in range(feature_length):
        column = Y_upsampled[:, 2 + feat]
        freq[feat] = np.sum(column)

    return X_upsampled, Y_upsampled


if __name__ == "__main__":

    gen_folder = "/home/tarask/Documents/Datasets/SaGa/Processed/feat/"
    dataset_name = subfolder = "train_n_val"
    feature_name = "R.S.Semantic Feature"
    feature_dim = 8

    create_dataset(gen_folder, subfolder, feature_name, dataset_name, feature_dim)