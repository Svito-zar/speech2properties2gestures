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
        feature_dim:        his defines how many values given feature can attain.
                            for example `phase` has feature_dim 4: {'iconic', 'discourse', 'deictic', 'beat'}

    Returns:
        Nothing, save dataset in npy file
    """

    curr_folder = general_folder + specific_subfolder + "/"

    # Initialize empty lists for the dataset input and output
    X_dataset = []
    Y_dataset = []

    # go though the dataset recordings
    for recording_id in range(1, 26):
        feat_file = str(recording_id).zfill(2) + "_feat.hdf5"

        # if this recording belong to the current dataset
        if feat_file in os.listdir(curr_folder):

            print("Consider file number :", str(recording_id).zfill(2))

            feat_hf = h5py.File(name=curr_folder + feat_file, mode='r')

            # check if the file contain the given feature
            spec_feat_hf = feat_hf.get(feature_name)
            if spec_feat_hf is None:
                continue

            # Obtain timing information
            text_file = str(recording_id).zfill(2) + "_text.hdf5"
            text_hf = h5py.File(name=curr_folder + text_file, mode='r')
            text_array = text_hf.get("text")
            word_starts = text_array[:, 0].round(1)
            word_ends = text_array[:, 1].round(1)

            # Consider all the time-frames except for the first and last three words, since we need three words for the contexts
            start_time = word_starts[3]
            end_time = (word_ends[-4] - 0.3).round(1)
            duration = (end_time - start_time).round(1)
            total_number_of_frames = int(duration * 5) + 1  # 0.2s time-steps

            curr_file_X_data = extract_text_from_the_current_file(text_hf, start_time, end_time, total_number_of_frames)

            spec_feat_hf = feat_hf.get(feature_name)
            curr_file_Y_data = extract_features_from_the_current_file(spec_feat_hf, recording_id, start_time,
                                                                            end_time, total_number_of_frames, feature_dim)

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

    # merge features
    Y_m = merge_feat(Y)

    # save files
    np.save(gen_folder + dataset_name+ "_Y_" + feature_name + ".npy", Y_m)
    np.save(gen_folder + dataset_name + "_X_" + feature_name + ".npy", X)


def merge_dir_n_relativ_pos(feature_array):
    """
    Merge several features which represent the same thing:
    "direction" and "relative position"

    'R.G.Right Semantic': {0: 'Amount', 1: 'Direction', 2: 'Shape', 3: 'relative Position', 4: 'Size'}
    'R.G.Left Semantic': {0: 'Amount', 1: 'Shape', 2: 'relative Position', 3: 'Size'}

    Args:
        feature_array:     {0: 'Amount', 1: 'Direction', 2: 'Shape', 3: 'relative Position', 4: 'Size'}

    Returns:
        new_feature_array: {0: 'Amount', 1: 'Shape', 2: 'relative Position', 3: 'Size'}

    """


    direction = np.clip(feature_array[:, 3] + feature_array[:, 5], 0, 1)
    feature_array[:, 5] = direction
    # remove "relative Position", which we already merged above
    new_feature_array = np.delete(feature_array, 3, 1)

    return new_feature_array



def extract_text_from_the_current_file(text_hf, start_time, end_time, total_number_of_frames):
    """
    Extract text features from a given file

    Args:
        text_hf:                hdf5 file with the transcript
        start_time:             start time
        end_time:               end time
        total_number_of_frames: total number of frames in the future feature file

    Returns:
        curr_file_X_data:       [total_number_of_frames, 7, 769] array of text features

    """

    text_array = text_hf.get("text")
    word_starts = text_array[:, 0].round(1)

    # First save all the text features
    curr_file_X_data = np.zeros((total_number_of_frames, 7, 769))

    time_ind = 0
    for time_st in np.arange(start_time, end_time - 0.1, 0.2):
        # find the corresponding words
        curr_word_id = bisect.bisect(word_starts, time_st)

        # encode current word with the next three and previous three words
        # while also storing time offset from the current time-step
        input_vector = [
            np.concatenate(([word_starts[word_id] - time_st], text_array[word_id, 2:]))
            for word_id in range(curr_word_id - 3, curr_word_id + 4)]

        curr_file_X_data[time_ind] = np.array(input_vector)

        time_ind += 1

    return curr_file_X_data


def extract_features_from_the_current_file(spec_feat_hf, recording_id, start_time, end_time, total_number_of_frames, feature_dim):
    """
    Extract given feature from a given file

    Args:
        spec_feat_hf:           hdf5 file for the given feature
        recording_id:           recording ID
        start_time:             start time
        end_time:               end time
        total_number_of_frames: total number of frames in the future feature file
        feature_dim:            dimensionality of the features

    Returns:
        curr_file_X_data:       [total_number_of_frames, n_features] array of features
    """

    if spec_feat_hf is None:
        return None

    spec_feat = np.array(spec_feat_hf)

    # Create dataset for Y features
    curr_file_Y_data = np.zeros((total_number_of_frames, feature_dim + 2))

    # Add timing info
    curr_file_Y_data[:, 1] = np.linspace(start_time, end_time, num=total_number_of_frames)

    for feat_id in range(spec_feat.shape[0]):

        curr_feat_vec = spec_feat[feat_id]

        # First two values contain st_time and end_time, other values - feature vector itself
        curr_feat_timing = curr_feat_vec[:2].round(1)
        curr_feat_values = curr_feat_vec[2:]

        for time_st in np.arange(curr_feat_timing[0], curr_feat_timing[1], 0.2):

            time_st = time_st.round(1)

            if time_st >= end_time:
                break

            # Save some extra info which might be useful later on
            output_vector = np.concatenate(([recording_id, time_st], curr_feat_values))

            time_ind = int((time_st - start_time) * 5)

            curr_file_Y_data[time_ind] = output_vector

    return curr_file_Y_data


def merge_feat(Y):
    """

    Args:
        Y:                  output dataset with binary gesture properties

    Returns:
        Y_train_n_val:      fixed input dataset with the features merged together

    """

    print(Y.shape)

    Y_train_n_val = Y

    # Merge certain features together
    direction = np.clip(Y_train_n_val[:, 3] + Y_train_n_val[:, 7], 0, 1)
    Y_train_n_val[:, 3] = direction
    # remove "relative Position", which we already merged above
    Y_train_n_val = np.delete(Y_train_n_val, 7, 1)
    # remove "Deictic"
    Y_train_n_val = np.delete(Y_train_n_val, 4, 1)
    print(np.array(Y_train_n_val).shape)

    return Y_train_n_val


if __name__ == "__main__":

    gen_folder = "/home/tarask/Documents/Datasets/SaGa/Processed/feat/"
    dataset_name = subfolder = "train_n_val"

    feature_dim = 8
    feature_name = "Semantic"
    feature_name = "R.S.Semantic Feature"

    create_dataset(gen_folder, subfolder, feature_name, dataset_name, feature_dim)