import h5py
import numpy as np
import os
import bisect
import pympi

from my_code.data_processing.tools import calculate_spectrogram, extract_prosodic_features


def correct_the_time(time_st):
    """
    Convert the time to ending with .2, .4, .6 .8, .0
    since we have time steps of 0.2

    Args:
        time_st: current time step

    Returns:
        time_st: fixed time step (if it needs to be fixed)

    """
    if int(time_st * 10) % 2 == 1:
        time_st += 0.1
    return round(time_st, 1)


def create_dataset(raw_data_folder, general_folder, specific_subfolder, feature_name, dataset_name, feature_dim):
    """

    Args:
        raw_data_folder:    folder where all the raw data is stored
        general_folder:     folder where all the used data is stored
        specific_subfolder: name of a specific subfolder
        feature_name:       name of the feature we are considering
        dataset_name:       name of the dataset
        feature_dim:        his defines how many values given feature can attain.
                            for example `phase` has feature_dim 4: {'iconic', 'discourse', 'deictic', 'beat'}

    Returns:
        Nothing, save dataset in npy file
    """

    curr_folder = general_folder + specific_subfolder + "/"

    context_length = 5

    # Initialize empty lists for the dataset input and output
    X_dataset = []
    Y_dataset = []
    A_dataset = []

    audio_dir = "/home/tarask/Documents/Datasets/SaGa/Raw/Audios/enhanced/"

    # go though the dataset recordings
    for recording_id in range(1, 26):
        feat_file = str(recording_id).zfill(2) + "_feat.hdf5"

        # if this recording belong to the current dataset
        if feat_file in os.listdir(curr_folder):

            print("\nConsider file number :", str(recording_id).zfill(2))

            feat_hf = h5py.File(name=curr_folder + feat_file, mode='r')

            if feature_name == "Semantic":
                right_feat_name = "R.G.Right " + feature_name
                left_feat_name = "R.G.Left " + feature_name
            elif feature_name == "Phrase" or feature_name == "Phase":
                right_feat_name = "R.G.Right." + feature_name
                left_feat_name = "R.G.Left." + feature_name
            elif feature_name == "R.S.Semantic Feature":
                right_feat_name = feature_name
                left_feat_name = "None"

            # check if the file contain the given feature
            spec_feat_hf = feat_hf.get(right_feat_name)
            if spec_feat_hf is None:
                spec_feat_hf = feat_hf.get(left_feat_name)
                if spec_feat_hf is None:
                    print("\nFile ", feat_file, " does not contain feature ", feature_name, " but only ", feat_hf.keys())
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
            # make sure that the time step fit in the general step
            start_time = correct_the_time(start_time)
            end_time = correct_the_time(end_time)
            # calculate duration
            duration = (end_time - start_time).round(1)
            total_number_of_frames = int(duration * 5) + 1  # 0.2s time-steps

            audio_file_name = audio_dir + "V" + str(recording_id) + "K3.mov_enhanced.wav"
            curr_file_audio_data = extract_audio_from_the_current_file(audio_file_name, start_time, end_time, total_number_of_frames, context_length)

            curr_file_text_data = extract_text_from_the_current_file(text_hf, start_time, end_time, total_number_of_frames)

            spec_feat_hf = feat_hf.get(right_feat_name)
            if feature_name == "Semantic":
                curr_file_Y_right_data = extract_features_from_the_current_file(spec_feat_hf, recording_id, start_time,
                                                                            end_time, total_number_of_frames, feature_dim+1)
                if spec_feat_hf is not None:
                    curr_file_Y_right_data = merge_dir_n_relativ_pos(curr_file_Y_right_data)
            else:
                curr_file_Y_right_data = extract_features_from_the_current_file(spec_feat_hf, recording_id, start_time,
                                                                                end_time, total_number_of_frames,
                                                                                feature_dim)

            spec_feat_hf = feat_hf.get(left_feat_name)
            curr_file_Y_left_data = extract_features_from_the_current_file(spec_feat_hf, recording_id, start_time,
                                                                            end_time, total_number_of_frames, feature_dim)

            # Merge both hands together
            if curr_file_Y_left_data is not None and curr_file_Y_right_data is not None:
                curr_file_Y_data = np.maximum(curr_file_Y_left_data, curr_file_Y_right_data)
            else:
                if curr_file_Y_left_data is not None:
                    curr_file_Y_data = curr_file_Y_left_data
                else:
                    curr_file_Y_data = curr_file_Y_right_data

            if feature_name == "Phrase":
                curr_file_Y_data = remove_redundant_phrases(curr_file_Y_data)

            # See if the time difference between the two time steps is always the same
            time_dif = (curr_file_Y_data[1:, 1] - curr_file_Y_data[:-1, 1]).round(1)
            max_td = np.max(time_dif)
            min_td = np.min(time_dif)
            if max_td != 0.2 or min_td != 0.2:
                print("WRONG TIMING IN : ", np.where(time_dif != 0.2))
            print("Time difference is in [", min_td, ", ", max_td, "]")

            # remove data when interlocutor speaks
            curr_file_A_data, curr_file_X_data, curr_file_Y_data = remove_data_when_interlocutors_speaks(curr_file_audio_data,
                                                                                       curr_file_text_data,
                                                                                       curr_file_Y_data, start_time,
                                                                                       end_time, recording_id,
                                                                                       raw_data_folder)

            if len(X_dataset) == 0:
                X_dataset = curr_file_X_data
                A_dataset = curr_file_A_data
                Y_dataset = curr_file_Y_data
            else:
                X_dataset = np.concatenate((X_dataset, curr_file_X_data))
                A_dataset = np.concatenate((A_dataset, curr_file_A_data))
                Y_dataset = np.concatenate((Y_dataset, curr_file_Y_data))

            print(np.asarray(X_dataset, dtype=np.float32).shape)
            print(np.asarray(A_dataset, dtype=np.float32).shape)
            print(np.asarray(Y_dataset, dtype=np.float32).shape)

            # ensure synchronization
            assert Y_dataset.shape[0] == A_dataset.shape[0] == X_dataset.shape[0]

    # create dataset file
    Audio_feat = np.asarray(A_dataset, dtype=np.float32)

    Y = np.asarray(Y_dataset, dtype=np.float32)
    X = np.asarray(X_dataset, dtype=np.float32)

    # merge features
    if feature_name == "R.S.Semantic Feature":
        Y = merge_sp_semantic_feat(Y)

    # calculate frequencies
    feat_dim = Y.shape[1] - 2
    freq = np.zeros(feat_dim)
    for feat in range(feat_dim):
        column = Y[:, 2 + feat]
        freq[feat] = np.sum(column)  # These are the binary gesture properties

    print("Frequencies are: ", freq)

    # save files
    np.save(gen_folder + dataset_name+ "_Y_" + feature_name + ".npy", Y)
    np.save(gen_folder + dataset_name + "_X_" + feature_name + ".npy", X)
    np.save(gen_folder + dataset_name + "_A_" + feature_name + ".npy", Audio_feat)


def remove_data_when_interlocutors_speaks(curr_file_audio_data, curr_file_text_data, curr_file_prop_data, record_start_time, record_end_time, recording_id, raw_data_folder):
    """
    Delete all the frames when interlocutor was speakers
    Args:
        curr_file_audio_data:     current file audio data
        curr_file_text_data:      current file text data
        curr_file_prop_data:      current file properties data
        record_start_time:        starting time of the current recording
        record_end_time:          ending time of the current recording
        recording_id:             recording ID
        raw_data_folder:          folder where all the raw data is stored

    Returns:
        curr_file_audio_data:     new current file audio data
        curr_file_prop_data:      new current file properties data
    """

    elan_file_name = raw_data_folder + str(recording_id).zfill(2) + "_video.eaf"
    elan = pympi.Elan.Eaf(file_path=elan_file_name)

    # Take the part of the speech that belongs to the interlocutor.
    if "F.S.Form" not in elan.tiers:
        return curr_file_audio_data, curr_file_text_data, curr_file_prop_data
    else:
        curr_tier = elan.tiers["F.S.Form"][0]

    time_key = elan.timeslots

    indices_to_delete = []

    for key, value in curr_tier.items():
        (st_t, end_t, word, _) = value

        if word is not None:

            # Only consider words which are clearly not back channels
            word = word.lstrip()
            if word != "" and word != " ":
                if word != "mhm" and word != "hm" and word != 'OK' and word != 'ja' and word != "ah" and word != "äh":

                    # convert ms to s
                    curr_word_st_time = round(time_key[st_t] / 1000, 1)
                    curr_word_end_time = round(time_key[end_t] / 1000, 1)

                    # make sure that the time step fit in the general step
                    curr_word_st_time = correct_the_time(curr_word_st_time)

                    if curr_word_st_time > record_end_time:
                        break

                    if curr_word_end_time < record_start_time:
                        continue

                    time_ind = int(((curr_word_st_time - record_start_time) * 5).round())

                    for time_st in np.arange(curr_word_st_time, curr_word_end_time, 0.2):

                        if time_st < record_start_time:
                            continue

                        if time_st > record_end_time:
                            continue

                        indices_to_delete.append(time_ind)

                        time_ind += 1

    # remove repetitions
    # they occur, because sometimes incrementing 0.2 above results in numerical difference:
    # [46.2, 46.8] becomes [46.2, 46.79999]
    indices_to_delete = np.unique(indices_to_delete)

    # Delete all the frames when interlocutor speaks
    curr_file_A_data = np.delete(curr_file_audio_data, indices_to_delete, 0)
    curr_file_X_data = np.delete(curr_file_text_data, indices_to_delete, 0)
    curr_file_Y_data = np.delete(curr_file_prop_data, indices_to_delete, 0)


    return curr_file_A_data, curr_file_X_data, curr_file_Y_data


def merge_sp_semantic_feat(Y):
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


def remove_redundant_phrases(feature_array):
    """
    Remove several feature values which are irrelevant

    'R.G.Phrase': {0: 'deictic', 1: 'beat', 2: 'move', 3: 'indexing', 4: 'iconic', 5: 'discourse', 6: 'unclear'},
    'R.G.Phrase': {0: 'deictic', 1: 'beat', 2: 'iconic', 3: 'discourse''},

    Args:
        feature_array:    {0: 'deictic', 1: 'beat', 2: 'move', 3: 'indexing', 4: 'iconic', 5: 'discourse', 6: 'unclear'}

    Returns:
        new_feature_array:  {0: 'deictic', 1: 'beat', 2: 'iconic', 3: 'discourse''},

    """

    # remove  2: 'move', 3: 'indexing', 6: 'unclear' which are not relevant for us
    new_feature_array = np.delete(feature_array, [4,5,8], 1)


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
        curr_word_id = bisect.bisect(word_starts, time_st) - 1

        # encode current word with the next three and previous three words
        # while also storing time offset from the current time-step
        input_vector = [
            np.concatenate(([word_starts[word_id] - time_st], text_array[word_id, 2:]))
            for word_id in range(curr_word_id - 3, curr_word_id + 4)]

        curr_file_X_data[time_ind] = np.array(input_vector)

        time_ind += 1

    return curr_file_X_data


def extract_audio_from_the_current_file(audio_file, start_time, end_time, total_number_of_frames, context_length):
    """
    Extract audio features from a given file

    Args:
        audio_file:             audio file
        start_time:             start time
        end_time:               end time
        total_number_of_frames: total number of frames in the future feature file
        context_length:         how many previous and next frames to consider

    Returns:
        curr_file_A_data:       [total_number_of_frames, X, Y] array of audio features

    """

    fps = 20

    start_time = start_time.round(1)
    end_time = end_time.round(1)

    print("Timing: [", start_time, ", ", end_time, "]")
    print("Number of frames: ", total_number_of_frames)

    fps = 5  # steps are 0.2s

    """
    # extract spectrogram for the whole audio file
    print("Calculating spectrogram ... ")
    audio_array = calculate_spectrogram(audio_file, fps)

    # reduce the fps from 20 to 5, so 4 times
    end = 4 * int(len(audio_array) / 4)
    audio_array = np.mean(audio_array[:end].reshape(-1, audio_array.shape[1], 4), 2)
    
    print("SPECTRORAM Audio array shape: ", audio_array.shape)

    """

    # extract prosodic features for the whole audio file
    print("Calculating prosodic features ... ")
    audio_array = extract_prosodic_features(audio_file)
    print("PROSODIC Audio array shape: ", audio_array.shape)

    # create a list of sequences with a fixed past and future context length ( overlap them to use data more efficiently)
    start_ind = int(start_time*fps)
    seq_step = 1  # overlap of sequences: 0.2s

    stop_ind = int(end_time*fps) + 1

    assert start_ind > context_length
    assert stop_ind < audio_array.shape[0]

    curr_file_A_data = np.array([audio_array[i - context_length: i + context_length+1]
                                    for i in range(start_ind, stop_ind, seq_step)])

    return curr_file_A_data


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

    # Add recording info
    curr_file_Y_data[:, 0] = recording_id
    # Add timing info
    curr_file_Y_data[:, 1] = np.linspace(start_time, end_time, num=total_number_of_frames)

    for feat_id in range(spec_feat.shape[0]):

        curr_feat_vec = spec_feat[feat_id]

        # First two values contain st_time and end_time, other values - feature vector itself
        curr_feat_timing = curr_feat_vec[:2].round(1)
        curr_feat_values = curr_feat_vec[2:]

        curr_feat_start_time = curr_feat_timing[0]

        # make sure that the time step fit in the general step
        curr_feat_start_time = correct_the_time(curr_feat_start_time)

        for time_st in np.arange(curr_feat_start_time, curr_feat_timing[1], 0.2):

            time_st = time_st.round(1)

            if time_st < start_time:
                continue

            if time_st >= end_time:
                break

            # Save some extra info which might be useful later on
            output_vector = np.concatenate(([recording_id, time_st], curr_feat_values))

            time_ind = int(((time_st - start_time) * 5).round())

            curr_file_Y_data[time_ind] = output_vector

    return curr_file_Y_data


def upsample(X, Y, n_features):
    """

    Args:
        X:                  input dataset with text features
        Y:                  output dataset with binary gesture properties
        n_features:         number of features in the dataset

    Returns:
        X_upsampled:        upsampled input dataset with equalized features frequencies
        Y_upsampled:        upsampled output dataset with equalized features frequencies

    """

    print(Y.shape)


    freq = np.zeros(n_features)
    for feat in range(n_features):
        column = Y[:, 2 + feat]
        freq[feat] = np.sum(column) # These are the binary gesture properties
        if freq[feat] < 100:
            freq[feat] = 10000

    print(freq)

    max_freq = np.max(freq)
    multipliers = [int(max_freq // freq[feat]) for feat in range(n_features)]

    print("Multipliers: ", multipliers)

    Y_upsampled = list(np.copy(Y))
    X_upsampled = list(np.copy(X))

    for frame_ind in range(Y.shape[0]):
        multipl_factor = 1
        for curr_feat in range(n_features):
            if Y[frame_ind, curr_feat+2] == 1: # first two numbers are containing extra info
                multipl_factor = max(multipl_factor, multipliers[curr_feat])
        if multipl_factor > 1:
            X_upsampled += [X[frame_ind]] * multipl_factor
            Y_upsampled += [Y[frame_ind]] * multipl_factor

    X_upsampled = np.asarray(X_upsampled, dtype=np.float32)
    Y_upsampled = np.asarray(Y_upsampled, dtype=np.float32)

    print(Y_upsampled.shape)

    freq = np.zeros(n_features)
    for feat in range(n_features):
        column = Y_upsampled[:, 2 + feat]
        freq[feat] = np.sum(column)

    print("Freq: ", freq)

    return X_upsampled, Y_upsampled


if __name__ == "__main__":
    raw_data_folder = "/home/tarask/Documents/Datasets/SaGa/Raw/All_the_transcripts/"
    gen_folder = "/home/tarask/Documents/Datasets/SaGa/Processed/feat/"
    dataset_name = subfolder = "train_n_val"

    feature_dim = 8
    feature_name = "R.S.Semantic Feature"
    create_dataset(raw_data_folder, gen_folder, subfolder, feature_name, dataset_name, feature_dim)

    feature_dim = 7
    feature_name = "Phrase"
    create_dataset(raw_data_folder, gen_folder, subfolder, feature_name, dataset_name, feature_dim)

    feature_dim = 5
    feature_name = "Phase"
    create_dataset(raw_data_folder, gen_folder, subfolder, feature_name, dataset_name, feature_dim)

    exit(0)

    feature_dim = 4
    feature_name = "Semantic"
    create_dataset(raw_data_folder, gen_folder, subfolder, feature_name, dataset_name, feature_dim)