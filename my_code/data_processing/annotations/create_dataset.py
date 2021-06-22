from tqdm import tqdm
from os.path import join, isfile, basename
from warnings import WarningMessage
import h5py
import numpy as np
import os
import bisect
import pympi

from my_code.data_processing.tools import calculate_spectrogram, extract_prosodic_features


class MissingDataException(Exception):
    pass

# TODO(RN): TEMPORARY
ROUNDING_ERROR_COUNT = 0

def open_property_data_for_both_hands(hdf5_dataset, property_name):

    # Get left/right hand property names
    if property_name == "Semantic":
        right_property_name = "R.G.Right " + property_name
        left_property_name  = "R.G.Left "  + property_name
    elif property_name == "Phrase" or property_name == "Phase":
        right_property_name = "R.G.Right." + property_name
        left_property_name  = "R.G.Left."  + property_name
    elif property_name == "R.S.Semantic Feature":
        right_property_name = property_name
        left_property_name  = None
    else:
        raise MissingDataException("Unexpected property: ", property_name)
    
    right_property = hdf5_dataset.get(right_property_name)
    left_property = None if left_property_name is None else hdf5_dataset.get(left_property_name)

    if right_property is not None:
        right_property = np.array(right_property) 
    if left_property is not None:
        left_property = np.array(left_property)

    return left_property, right_property

def timestep_to_frame_index(timestep, start_time):
    """
    Convert the given timestep from seconds to the index of the corresponding frame.
    """
    frame_ind = round((timestep - start_time) * 5)
    prev_ind = round((timestep - 0.2 - start_time) * 5)
    # TEMP: count incorrect rounding issues
    if frame_ind == prev_ind:
        global ROUNDING_ERROR_COUNT
        ROUNDING_ERROR_COUNT += 1

    return round((timestep - start_time) * 5)

def correct_the_time(time_st):
    """
    Convert the time to ending with .2, .4, .6 .8, .0
    since we have time steps of 0.2

    Args:
        time_st: current time step

    Returns:
        time_st: fixed time step (if it needs to be fixed)

    """
    if round(time_st * 10) % 2 == 1:
        time_st += 0.1
    return round(time_st, 1)

def create_dataset(properties_to_consider, property_dims):
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
    gest_prop_dir = "/home/work/Desktop/repositories/probabilistic-gesticulator/my_code/data_processing/annotations/feat/gesture_properties"
    text_vec_dir  = "/home/work/Desktop/repositories/probabilistic-gesticulator/my_code/data_processing/annotations/feat/text_vectors"
    audio_dir = "/home/work/Desktop/repositories/probabilistic-gesticulator/my_code/data_processing/annotations/renamed_audio"
    
    recording_idx_progress_bar = tqdm(range(1, 26))
    for recording_idx in recording_idx_progress_bar:
        recording_idx = str(recording_idx).zfill(2)
        recording_idx_progress_bar.set_description(f"Recording {recording_idx}")
       
        audio_file = join(audio_dir, f"V{recording_idx}.mov_enhanced.wav")
        text_file = join(text_vec_dir, f"{recording_idx}_text.hdf5")
        gest_prop_file = join(gest_prop_dir, f"{recording_idx}_feat.hdf5")
        input_files = [audio_file, text_file, gest_prop_file]
        
        # The annotations are missing for some of the recordings
        missing_files = [basename(file) for file in input_files if not isfile(file)]
        if len(missing_files) > 0:
            print(f"Skipping recording {recording_idx} because of missing files: {missing_files}.")
            continue

        text_dataset = h5py.File(text_file, mode='r').get("text")
        word_starts = text_dataset[:, 0].round(1)
        word_ends = text_dataset[:, 1].round(1)

        start_time = correct_the_time(word_starts[3])
        end_time   = correct_the_time(word_ends[-4] - 0.3) # TODO(RN) why 0.3?
        duration   = correct_the_time(end_time - start_time)
        # NOTE: we have 0.2s timesteps
        total_number_of_frames = int(duration * 5) + 1


        text_features = extract_text_features(text_dataset, word_starts, start_time, end_time, total_number_of_frames)
        audio_features = extract_audio_features(audio_file, start_time, end_time, total_number_of_frames)
        
        gest_prop_hf = h5py.File(gest_prop_file, mode='r')

        progress_bar = zip(properties_to_consider, property_dims)
        for property_name, property_dim in progress_bar:
            # progress_bar.set_description(property_name)
            try:
                gest_prop_features = extract_gesture_property_features(gest_prop_hf, recording_idx, property_name, property_dim, start_time, end_time, total_number_of_frames)
                print("# rounding errors:", ROUNDING_ERROR_COUNT)
            except MissingDataException as warning_msg:
                print("WARNING:", warning_msg, "skipping to next file!")
                continue
    # -------------------------------------------------
    # TODO(RN) remaining stuff from old implementation:
    # -------------------------------------------------
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

def extract_gesture_property_features(
    hdf5_dataset, recording_idx, property_name, property_dim,
    recording_start_time, recording_end_time, total_number_of_frames
):
    """
    Transform the given hdf5 dataset of binary gesture property vectors into
    an array of shape (total_number_of_frames, 2 + property_dim).

    NOTE: The output array contains the 'recording_idx' and the timestep as extra information.
    NOTE: The features vectors of the left and the right hand are merged together.

    Args:
        hdf5_dataset:           a hdf5 dataset created with 'extract_binary_features.py'
        recording_idx:          the index of the recording that the gestures belong to
        property_name:          the name of the gesture property (one of "Phrase", "Semantic", "Phase" and "R.S.Semantic Feature")
        property_dim:           the dimensionality of the binary property vectors
        recording_start_time:   the start of the recording
        recording_end_time:     the end of the recording
        total_number_of_frames: the total number of frames in the output array
    
    Returns:
        The property dataset as an array of shape (total_number_of_frames, 2 + property_dim)
    """
    left_property_data, right_property_data = open_property_data_for_both_hands(hdf5_dataset, property_name)

    if left_property_data is None and right_property_data is None:
        raise MissingDataException(f"the '{property_name}' is missing")
    
    if property_name == "Semantic" and right_property_data is not None:
        right_property_data = merge_redundant_semantic_labels(right_property_data)

    # NOTE: we increase dim by 2 because of the two extra elements below

    left_feature_vectors = np.zeros((total_number_of_frames, 2 + property_dim))
    right_feature_vectors = np.zeros((total_number_of_frames, 2 + property_dim))

    
    # Add recording info
    left_feature_vectors[:, 0] = recording_idx
    right_feature_vectors[:, 0] = recording_idx

    # Add timing info
    
    timesteps = np.linspace(recording_start_time, recording_end_time, num=total_number_of_frames)
    # Correct numerical errors TODO(RN): it sucks that this is necessary
    timesteps = [correct_the_time(step) for step in timesteps]
    # TODO(RN) TEMPORARY: check rounding issues
    for i in range(1, len(timesteps)-1):
         if (timesteps[i] - timesteps[i-1]).round(1) != 0.2 or (timesteps[i+1] - timesteps[i]).round(1) != 0.2:
            print(timesteps[i-1 : i+2])
            exit()

    left_feature_vectors[:, 1] = timesteps
    right_feature_vectors[:, 1] = timesteps
    # Process the two hands separately    
    for properties, output_vectors in [
        [left_property_data, left_feature_vectors],
        [right_property_data, right_feature_vectors]
    ]:
        # One hand may be missing
        if properties is None:
            continue

        # Create the output feature vectors for the current hand
        for annotation_entry in properties:
            annotation_start_time = correct_the_time(annotation_entry[0])
            annotation_end_time   = correct_the_time(annotation_entry[1])
            binary_property_vector = annotation_entry[2:]
            
            for time_st in np.arange(annotation_start_time, annotation_end_time, 0.2):
                # Avoid numerical errors (e.g. 0.4 + 0.2 = 0.6000000000000001)
                time_st = time_st.round(1)
                
                if time_st < recording_start_time:
                    continue
                if time_st >= recording_end_time:
                    break
         
                frame_ind = timestep_to_frame_index(time_st, recording_start_time)
                feature_vector = np.concatenate(([recording_idx, time_st], binary_property_vector))
                output_vectors[frame_ind] = feature_vector
    
    # Merge the two hands together
    merged_feature_vectors = np.maximum(left_feature_vectors, right_feature_vectors)
    def check_time_dif(arr):
        # See if the time difference between the two time steps is always the same
        time_dif = (arr[1:, 1] - arr[:-1, 1]).round(1)
        max_td = np.max(time_dif)
        min_td = np.min(time_dif)
        if max_td != 0.2 or min_td != 0.2:
            print("WARNING: WRONG TIMING, time difference is in [", min_td, ", ", max_td, "]")
            print( [arr[int(i)-1:int(i)+2, 1] for i in np.argwhere(time_dif != 0.2) ] )

    check_time_dif(merged_feature_vectors)
    
    return merged_feature_vectors
    
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
    Merge certain speech semantic features (which are very similar or duplicate)

    'R.S.Semantic Feature': {0: 'Amount', 1: 'Direction', 2: 'Deictic', 3: 'Shape', 4: 'Property', 5: 'relative Position', 6: 'Size', 7: 'Entity'}}
    'R.S.Semantic Feature': {0: 'Amount', 1: 'Direction', 2: 'Shape', 3: 'Property', 4: 'Size', 5: 'Entity'}}

    Args:
        Y:                  output dataset with binary gesture properties

    Returns:
        Y_train_n_val:      fixed output dataset with the features merged together

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


def merge_redundant_semantic_labels(feature_array):
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
    feature_array[:, 5] = np.clip(feature_array[:, 3] + feature_array[:, 5], 0, 1)
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

def extract_text_features(text_dataset, word_starts, start_time, end_time, total_number_of_frames):
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
    sequence_length = 7
    n_bert_dims = 768
    # NOTE: 
    text_features = np.zeros((total_number_of_frames, sequence_length, 1 + n_bert_dims))
    time_ind = 0
    for timestep in tqdm(np.arange(start_time, end_time - 0.1, 0.2), desc="text", leave=False): # TODO(RN) why - 0.1?
        curr_word_idx = bisect.bisect(word_starts, timestep) - 1
        
        # TODO(RN): The time offset is negative in the original impl.,
        #           I think it should be the other way around
        feature_vector = [
            np.array( [word_starts[word_idx] - timestep] + list(text_dataset[word_idx, 2:]) )
            for word_idx in range(curr_word_idx - 3, curr_word_idx + 4)]
        
        text_features[time_ind] = np.array(feature_vector)
        time_ind += 1

    return text_features

def extract_audio_features(audio_file, start_time, end_time, total_number_of_frames):
    """
    Extract audio features from a given file

    Args:
        audio_file:             audio file
        start_time:             start time
        end_time:               end time
        total_number_of_frames: total number of frames in the future feature file

    Returns:
        curr_file_A_data:       [total_number_of_frames, X, Y] array of audio features

    """
    fps = 5
    context_length = 5
    # print("Timing: [", start_time, ", ", end_time, "]")
    # print("Number of frames: ", total_number_of_frames)

    prosodic_features = extract_prosodic_features(audio_file)
    # print("PROSODIC Audio array shape: ", prosodic_features.shape)

    # create a list of sequences with a fixed past and future context length ( overlap them to use data more efficiently)
    start_ind = int(start_time*fps)
    seq_step = 1  # overlap of sequences: 0.2s

    stop_ind = int(end_time*fps) + 1

    assert start_ind > context_length
    assert stop_ind < prosodic_features.shape[0]

    audio_features = np.array([
        prosodic_features[i - context_length : i + context_length + 1]
        for i in range(start_ind, stop_ind, seq_step)])

    return audio_features

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

    # upsample
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
    # raw
    ELAN_TRANSCRIPT_DIR = "/home/work/Desktop/repositories/probabilistic-gesticulator/dataset/All_the_transcripts"
    # gen
    GESTURE_PROPERTY_VECTORS_DIR = "/home/work/Desktop/repositories/probabilistic-gesticulator/my_code/data_processing/annotations/feat"
    

    # currently we extract features for one property at a time, but it could be significantly
    # improved, since input features are always the same and don't need to be recalculated

    feature_dims = [8, 7, 5, 4]
    feature_names = ["R.S.Semantic Feature", "Phrase", "Phase", "Semantic"]
    create_dataset(feature_names, feature_dims)
