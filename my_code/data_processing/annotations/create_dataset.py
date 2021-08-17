from tqdm import tqdm
from os.path import join, isfile, basename
import h5py
import numpy as np
import os
import bisect
import pympi

from my_code.data_processing.tools import calculate_spectrogram, extract_prosodic_features


class MissingDataException(Exception):
    pass

def print_shape(name, data):
    """Print the name and the shape of the dataset."""
    tqdm.write(f"{name}: {data.shape}")

def check_time_diff(arr):
    # See if the time difference between the two time steps is always the same
    time_dif = (arr[1:, 1] - arr[:-1, 1]).round(1)
    max_td = np.max(time_dif)
    min_td = np.min(time_dif)
    if max_td != 0.2 or min_td != 0.2:
        tqdm.write("WARNING: WRONG TIMING, time difference is in [", min_td, ", ", max_td, "]")
        tqdm.write( [arr[int(i)-1:int(i)+2, 1] for i in np.argwhere(time_dif != 0.2) ] )

def get_timesteps_between(start_time, end_time):
    """
    Get a list of 0.2s timesteps between the two given timestamps, with 'end_time' included.
    """
    start_time = correct_the_time(start_time)
    end_time = correct_the_time(end_time)
    n_frames = calculate_number_of_frames(start_time, end_time)
    timesteps = np.linspace(start_time, end_time, num=n_frames, endpoint=False)
    # Correct numerical errors 
    timesteps = [correct_the_time(step) for step in timesteps] 

    return timesteps

def calculate_number_of_frames(start_time, end_time):
    """
    Return the number of frames between the two timestamps, assuming 5 FPS.
    """
    duration = correct_the_time(end_time - start_time)
    # NOTE: we have 0.2s timesteps
    n_frames = int(duration * 5)

    return n_frames

def get_left_right_property_name(property_name):
    if property_name == "Semantic":
        R_property_name = "R.G.Right " + property_name
        L_property_name = "R.G.Left "  + property_name

    elif property_name == "Phrase" or property_name == "Phase":
        R_property_name = "R.G.Right." + property_name
        L_property_name = "R.G.Left."  + property_name

    elif property_name == "S_Semantic":
        # Speech semantics are not related to the hands, so we arbitrarily store them in the right hand
        R_property_name = "R.S.Semantic Feature"
        L_property_name = " "

    else:
        raise MissingDataException("Unexpected property: ", property_name)

    return L_property_name, R_property_name

def open_and_clean_property_data_for_both_hands(hdf5_dataset, property_name):
    """
    Return the left/right gesture property arrays from the given dataset,
    with unwanted labels merged or removed.
    """
    # Get left/right hand property names
    L_property_name, R_property_name = get_left_right_property_name(property_name)
    
    # Open the property arrays in the hdf5 dataset
    L_property_data = np.array(hdf5_dataset.get(L_property_name))
    R_property_data = np.array(hdf5_dataset.get(R_property_name))

    # Clean up the property labels
    if property_name == "Semantic":
        if not is_empty(R_property_data):
            # Only the right hand labels need to be cleaned
            R_property_data = merge_redundant_gesture_semantic_labels(R_property_data)
    
    elif property_name == "Phrase":
        L_property_data = remove_unwanted_phrase_labels(L_property_data)
        R_property_data = remove_unwanted_phrase_labels(R_property_data)

    elif property_name == "S_Semantic":
        # Speech semantics are stored in the "right hand"
        if not is_empty(R_property_data):
            R_property_data = merge_redundant_speech_semantic_labels(R_property_data)

    elif property_name != "Phase":
        raise ValueError("Unexpected property: ", property_name)

    return L_property_data, R_property_data

def timestep_to_frame_index(timestep, start_time):
    """
    Convert the given timestep from seconds to the index of the corresponding frame.
    """
   
    frame_ind = round((timestep - start_time) * 5)
    prev_ind = round((timestep - 0.2 - start_time) * 5)

    return round((timestep - start_time) * 5)

def correct_the_time(time_st):
    """
    Round the given timestep to the a multiple of 0.2,
    since we have time steps of 0.2 seconds.
    """
    if round(time_st * 10) % 2 == 1:
        time_st += 0.1
    return round(time_st, 1)

def create_datasets(audio_dir, text_dir, gest_prop_dir, elan_dir, property_names, property_dims, output_dir, held_out_idxs):
    """
    Create np.array datasets containing aligned audio, text and binary gesture property frames.
    
    Args:
        audio_dir:      the folder with the audio files
        text_dir:       the folder with the text transcriptions (see 'encode_text.py')
        gest_prop_dir:  the folder containing the property vectors (see 'extract_binary_features.py')
        property_names: a list of gesture property names 
        property_dims:  a list of the corresponding dimensionalities
        output_dir:     the folder where the datasets will be saved
    Returns:
        nothing, but it saves the audio/text/property arrays into 'output_dir' per file

    NOTE: see 'open_and_clean_property_data_for_both_hands()' for the list of supported properties
    NOTE: each frame is 0.2 seconds long.
    """
    all_audio_features     = []
    all_text_features      = []
    all_gest_prop_features = []
    
    # The speech semantic properties are missing for some recordings. Therefore we
    # will save the audio and text separately, without the missing files, for the
    # speech semantic property.
    idxs_without_speech_semantics = []

    recording_idx_progress_bar = tqdm(range(1, 26))
    for recording_idx in recording_idx_progress_bar:
        if recording_idx in held_out_idxs:
            tqdm.write('-'*40)
            tqdm.write(f"Recording {recording_idx} is held out, skipping to next file.")
            tqdm.write('-'*40)
            continue
        # Update progress par
        recording_idx = str(recording_idx).zfill(2)
        recording_idx_progress_bar.set_description(f"Recording {recording_idx}")
       
        # Check for missing files
        audio_file     = join(audio_dir, f"V{str(int(recording_idx))}K3.mov_enhanced.wav")
        text_file      = join(text_dir, f"{recording_idx}_text.hdf5")
        gest_prop_file = join(gest_prop_dir, f"{recording_idx}_feat.hdf5")
        input_files    = [audio_file, text_file, gest_prop_file]
        missing_files = [basename(file) for file in input_files if not isfile(file)]
        if len(missing_files) > 0:
            tqdm.write('-'*40)
            tqdm.write(f"WARNING: Skipping recording {recording_idx} because of missing files: {missing_files}.")
            tqdm.write('-'*40)
            continue
            
        # Open the encoded hdf5 datasets
        gest_prop_hf = h5py.File(gest_prop_file, mode='r')
        text_vec_hf = h5py.File(text_file, mode='r')
        text_dataset = text_vec_hf.get("text")
        # text_vec_hf.close()
        
        # Extract timing info from the dataset
        word_starts             = text_dataset[:, 0].round(1)
        word_ends               = text_dataset[:, 1].round(1)
        # We reserve first and last three words as context
        recording_start_time    = correct_the_time(word_starts[3])
        recording_end_time      = correct_the_time(word_ends[-4])
        total_number_of_frames  = calculate_number_of_frames(
            recording_start_time, recording_end_time)

        audio_features = extract_audio_features(
            audio_file, 
            recording_start_time, recording_end_time, 
            total_number_of_frames, elan_dir,  f"{recording_idx}_video"
        )

        text_features = extract_text_features(
            text_dataset, word_starts, 
            recording_start_time, recording_end_time, 
            total_number_of_frames
        )

        gest_prop_features = create_gesture_property_datasets(
            recording_idx, gest_prop_hf, 
            property_names, property_dims, 
            recording_start_time, recording_end_time, 
            total_number_of_frames
        )

        for property_dataset in gest_prop_features:
            if is_empty(property_dataset):
                # Store array index if speech semantic property is missing from current file
                idxs_without_speech_semantics.append(len(all_audio_features))
            else:
                # Ensure that the frames are aligned
                assert len(property_dataset) == len(audio_features) == len(text_features) or len(property_dataset) == 0
     
        all_audio_features.append(audio_features)
        all_text_features.append(text_features)
        all_gest_prop_features.append(gest_prop_features)

        gest_prop_hf.close()
        
    final_audio_dataset = np.concatenate(all_audio_features)
    final_text_dataset = np.concatenate(all_text_features)
        
        

    np.save(join(output_dir, "Audio.npy"), final_audio_dataset)
    np.save(join(output_dir, "Text.npy"), final_text_dataset)
    
    tqdm.write("-"*80 + "\nFinal dataset shapes:\n" + "-"*80)
    print_shape("audio", final_audio_dataset)
    print_shape("text", final_text_dataset)
   
    save_property_datasets(all_gest_prop_features, property_names)
    
    # Separately save the audio/text from those files which have the speech semantic property
    n_files = len(all_audio_features)
    
    kept_audio = [all_audio_features[idx] for idx in range(n_files) if idx not in idxs_without_speech_semantics]
    kept_text  = [all_text_features[idx]  for idx in range(n_files) if idx not in idxs_without_speech_semantics]
    
    s_semantic_audio = np.concatenate(kept_audio)
    s_semantic_text = np.concatenate(kept_text)
    np.save(join(output_dir, "S_Semantic_Audio.npy"), s_semantic_audio)
    np.save(join(output_dir, "S_Semantic_Text.npy"), s_semantic_text)
    
    print_shape("S_Semantic audio", s_semantic_audio)
    print_shape("S_Semantic_text", s_semantic_text)
    
def save_property_datasets(all_gest_prop_features, property_names):

    for property_idx, property_name in enumerate(property_names):
        curr_property_features = []
        for file_properties in all_gest_prop_features:
            if not is_empty(file_properties[property_idx]):
                curr_property_features.append(file_properties[property_idx])
        
        output_data = np.concatenate(curr_property_features)
        output_file = join(output_dir, f"{property_name}_properties.npy")
        np.save(output_file, output_data)
        print_shape(property_name + " labels", output_data)


def create_gesture_property_datasets(
    recording_idx, hdf5_dataset, property_names, property_dims,
    recording_start_time, recording_end_time, total_number_of_frames
):
    """
    Transform the given hdf5 dataset of binary gesture property vectors into
    arrays with shape (total_number_of_frames, 2 + property_dim).

    NOTE: See 'open_and_clean_property_data_for_both_hands()' for a list of supported properties.
    NOTE: The feature arrays contain the 'recording_idx' and the timestep as extra information.
    NOTE: The features vectors of the left and the right hand are merged together in each frame.
    
    Args:
        hdf5_dataset:           a hdf5 dataset created with 'extract_binary_features.py'
        recording_idx:          the index of the recording that the gestures belong to
        property_names:         the names of the considered gesture properties
        property_dims:          the corresponding property dimensionalities
        recording_start_time:   the start of the recording (in seconds)
        recording_end_time:     the end of the recording (in seconds)
        total_number_of_frames: the total number of frames in the output array
    
    Returns:
        A list of array datasets, one for each property, with shapes (total_number_of_frames, 2 + property_dim).
    """
    dataset_list = []
    for property_name, property_dim in zip(property_names, property_dims):
        try:
            features = _extract_gesture_property_features(
                recording_idx, hdf5_dataset, property_name, property_dim,
                recording_start_time, recording_end_time, total_number_of_frames
            )
            dataset_list.append(features)
        except MissingDataException as warning_msg:
            tqdm.write('-'*40)
            tqdm.write(f"WARNING: {warning_msg}, skipping to next property.")
            tqdm.write('-'*40)
            dataset_list.append(np.array(None))
            continue

        assert features.shape == (total_number_of_frames, property_dim+2)

    return dataset_list

def is_empty(array):
    return array.shape == ()

def _extract_gesture_property_features(
    recording_idx, hdf5_dataset, 
    property_name, property_dim,
    recording_start_time, recording_end_time, 
    total_number_of_frames
):
    """
    Create the gesture property dataset array for the given property 
    with a shape of (total_number_of_frames, property_dim + 2).
    
    NOTE: The two hands are merged together.
    """
    L_property_data, R_property_data = open_and_clean_property_data_for_both_hands(hdf5_dataset, property_name)

    if is_empty(L_property_data) and is_empty(R_property_data):
        raise MissingDataException(f"the '{property_name}' property is missing")
    
    L_feature_vectors = np.zeros((total_number_of_frames, 2 + property_dim))
    R_feature_vectors = np.zeros((total_number_of_frames, 2 + property_dim))

    # Add recording info
    L_feature_vectors[:, 0] = recording_idx
    R_feature_vectors[:, 0] = recording_idx

    # Add timing info
    timesteps = get_timesteps_between(start_time=recording_start_time, end_time=recording_end_time)
    
    L_feature_vectors[:, 1] = timesteps
    R_feature_vectors[:, 1] = timesteps
    
    #----------------------------------------------------
    # TODO(RN) TEMPORARY: check rounding issues
    for i in range(1, len(timesteps)-1):
         if (timesteps[i] - timesteps[i-1]).round(1) != 0.2 or (timesteps[i+1] - timesteps[i]).round(1) != 0.2:
            tqdm.write(timesteps[i-1 : i+2])
            raise ValueError("rounding_issues")
    #----------------------------------------------------

    # Process the two hands separately    
    for properties, output_vectors in [
        [L_property_data, L_feature_vectors],
        [R_property_data, R_feature_vectors]
    ]:
        # One hand may be missing
        if is_empty(properties):
            continue

        # Create the feature vectors for the current hand
        for annotation_entry in properties:
            annotation_start_time = correct_the_time(annotation_entry[0])
            annotation_end_time   = correct_the_time(annotation_entry[1])
            binary_property_vector = annotation_entry[2:]
            
            for time_st in get_timesteps_between(annotation_start_time, annotation_end_time):                
                if time_st < recording_start_time:
                    continue
                if time_st >= recording_end_time:
                    break

                # Store the feature vector in the corresponding frame
                frame_ind = timestep_to_frame_index(time_st, recording_start_time )
                feature_vector = np.concatenate(([recording_idx, time_st], binary_property_vector))
                output_vectors[frame_ind] = feature_vector
    
    # Merge the two hands together
    merged_feature_vectors = np.maximum(L_feature_vectors, R_feature_vectors)

    # Ensure that the timesteps are correct 
    check_time_diff(merged_feature_vectors)
    
    return merged_feature_vectors
    
def remove_data_when_interlocutor_speaks(
    elan_dir, audio_features, text_features, list_of_gest_prop_features,
    recording_idx, recording_start_time, recording_end_time
):
    """
    """
    # Open the ELAN annotation file
    elan_file_name = join(elan_dir, f"{recording_idx}_video.eaf")
    elan = pympi.Elan.Eaf(file_path=elan_file_name)

    # Interlocutor annotations might be missing
    if "F.S.Form" not in elan.tiers:
        return audio_features, text_features, list_of_gest_prop_features

    timeslots = elan.timeslots
    interlocutor_annotations = elan.tiers["F.S.Form"][0]

    indices_to_delete = []

    for annotation_entry in interlocutor_annotations.values():
        st_t, end_t, word, _ = annotation_entry
        word = word.strip()
        
        # Only consider words which are clearly not back channels
        if word == "" or word in ["mhm", "hm", "OK", "ja", "ah", "äh"]:
            continue
            
        # Convert ms to s and make sure the timestep fits with the 0.2 sec frames
        word_start_time = correct_the_time(timeslots[st_t] / 1000)
        word_end_time = correct_the_time(timeslots[end_t] / 1000)

        if word_start_time > recording_end_time:
            break

        if word_end_time < recording_start_time:
            continue
        
        timesteps = get_timesteps_between(word_start_time, word_end_time)
        for time_st in timesteps:           
            frame_ind = timestep_to_frame_index(time_st, recording_start_time)
            indices_to_delete.append(frame_ind)

    # Make sure that all indices are unique
    assert(np.array(np.array_equal(indices_to_delete, np.unique(indices_to_delete))))
    
    # Delete the selected frames
    n_frames = len(indices_to_delete)
    n_seconds = round(n_frames / 5)
    tqdm.write(f"Recording {recording_idx}:", end="\t")
    tqdm.write(f"INFO: Deleting {n_frames:<4} frames (~{n_seconds:<3} seconds) where the interlocutor was speaking.")


    def remove_data(array):
        if is_empty(array):
            tqdm.write("INFO: Detected missing gesture property.")
        else:
            array = np.delete(array, indices_to_delete, axis=0)
        return array
    
    audio_features = remove_data(audio_features)
    text_features = remove_data(text_features)
    list_of_gest_prop_features = [remove_data(feats) for feats in list_of_gest_prop_features]
    
    return audio_features, text_features, list_of_gest_prop_features


def merge_redundant_speech_semantic_labels(Y):
    """
    Merge certain speech semantic features (which are very similar or duplicate)

    'R.S.Semantic Feature': {0: 'Amount', 1: 'Direction', 2: 'Deictic', 3: 'Shape', 4: 'Property', 5: 'relative Position', 6: 'Size', 7: 'Entity'}}
    'R.S.Semantic Feature': {0: 'Amount', 1: 'Direction', 2: 'Shape', 3: 'Property', 4: 'Size', 5: 'Entity'}}

    Args:
        Y:                  output dataset with binary gesture properties

    Returns:
        Y_train_n_val:      fixed output dataset with the features merged together

    """


    Y_train_n_val = Y

    # Merge certain features together
    direction = np.clip(Y_train_n_val[:, 3] + Y_train_n_val[:, 7], 0, 1)
    Y_train_n_val[:, 3] = direction
    # remove "relative Position", which we already merged above
    Y_train_n_val = np.delete(Y_train_n_val, 7, 1)
    # remove "Deictic"
    Y_train_n_val = np.delete(Y_train_n_val, 4, 1)
   

    return Y_train_n_val


def merge_redundant_gesture_semantic_labels(feature_array):
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


def remove_unwanted_phrase_labels(feature_array):
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
    for timestep in tqdm(get_timesteps_between(start_time, end_time), desc="Processing text", leave=False):
        curr_word_idx = bisect.bisect(word_starts, timestep) - 1

        feature_vector = [
            np.array( [word_starts[word_idx] - timestep] + list(text_dataset[word_idx, 2:]) )
            for word_idx in range(curr_word_idx - 3, curr_word_idx + 4)]
        
        text_features[time_ind] = np.array(feature_vector)
        time_ind += 1

    return text_features

def extract_audio_features(audio_file, start_time, end_time, total_number_of_frames, elan_source_dir, fname):
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
    # tqdm.write("Timing: [", start_time, ", ", end_time, "]")
    # tqdm.write("Number of frames: ", total_number_of_frames)

    prosodic_features = extract_prosodic_features(audio_file)

    # create a list of sequences with a fixed past and future context length ( overlap them to use data more efficiently)
    start_ind = int(start_time*fps)
    seq_step = 1  # overlap of sequences: 0.2s

    stop_ind = int(end_time*fps)

    assert start_ind > context_length
    assert stop_ind < prosodic_features.shape[0]

    # remove interloculor's speech
    prosodic_features = mask_interlocutor_speech(prosodic_features, fps, elan_source_dir, fname)

    audio_features = np.array([
        prosodic_features[i - context_length : i + context_length + 1]
        for i in range(start_ind, stop_ind, seq_step)])

    assert(len(audio_features) == total_number_of_frames)

    return audio_features


def mask_interlocutor_speech(audio_feat_vectors, fps, elan_dir, elan_ann_fname):
    """
    Code taken from https://github.com/nagyrajmund/StyleGestures/blob/0892816b8f1e9b8980bbd42d04c1c665d2f7fdb4/data_processing/feature_extraction.py#L171

    Set every frame where the interlocutor is talking in the given audio_feat_vectors
    to zero using the transcription in the ELAN file in `elan_dir`.

    Args:
        audio_feat_vectors:        [T, D] - audio feature vector
        fps:                       int - frames per second (fps) rate
        elan_dir:                  str - directory with all the annotation ELAN files
        elan_ann_fname:            str - file name of the ELAN file to use

    Returns:
        audio_feat_vectors:        [T, D] - audio feature vector, where interlocutor speech is masked
    """

    # Define "silence" feature vector
    silence_vectors = extract_prosodic_features(join(os.getcwd(), "silence.wav"))
    audio_mask_feat_vec = silence_vectors[0]

    # Open the annotation file
    elan_file = join(elan_dir, elan_ann_fname + ".eaf")
    elan = pympi.Elan.Eaf(file_path=elan_file)

    if "F.S.Form" not in elan.tiers:
        tqdm.write("INFO: Interlocutor speech data is not present in current file.")
        return audio_feat_vectors

    def to_idx(timeslot):
        """
        Convert the given ELAN timestamp to the correspoding
        index of the audio_feat_vectors vector.
        NOTE: ELAN times are in milliseconds.
        """
        return elan.timeslots[timeslot] * fps // 1000

    n_masked = 0
    interlocutor_annotations = elan.tiers["F.S.Form"][0]
    for annotation_entry in interlocutor_annotations.values():
        start, end, word, _ = annotation_entry
        word = word.strip().lower()

        # Only consider words which are clearly not back channels
        if word in [
            "",
            "mhm",
            "hm",
            "ok",
            "ja",
            "ah",
            "äh",
            "aha",
        ]:
            continue

        # the end frame is not masked now, while I would rather mask it
        audio_feat_vectors[to_idx(start): to_idx(end)] = audio_mask_feat_vec
        n_masked += to_idx(end) - to_idx(start)

    tqdm.write(
        f"INFO: ({elan_ann_fname}) masking {n_masked / len(audio_feat_vectors) : .0%} of audio frames with zero due to interlocutor speech."
    )
    return audio_feat_vectors


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

    tqdm.write(Y.shape)


    freq = np.zeros(n_features)
    for feat in range(n_features):
        column = Y[:, 2 + feat]
        freq[feat] = np.sum(column) # These are the binary gesture properties
        if freq[feat] < 100:
            freq[feat] = 10000

    tqdm.write(freq)

    max_freq = np.max(freq)
    multipliers = [int(max_freq // freq[feat]) for feat in range(n_features)]

    tqdm.write("Multipliers: ", multipliers)

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

    tqdm.write(Y_upsampled.shape)

    freq = np.zeros(n_features)
    for feat in range(n_features):
        column = Y_upsampled[:, 2 + feat]
        freq[feat] = np.sum(column)

    tqdm.write("Freq: ", freq)

    return X_upsampled, Y_upsampled


if __name__ == "__main__":

    core_dir = "/home/tarask/Documents/Code/probabilistic-gesticulator/dataset/"
    gest_prop_dir = core_dir + "processed/gesture_properties/train_n_val/"
    text_vec_dir  = core_dir + "processed/word_vectors/train_n_val/"
    audio_dir     = core_dir + "audio/"
    elan_dir      = core_dir + "transcripts/"
    output_dir    = core_dir + "processed/numpy_arrays/train_n_val/"    
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    feature_dims = [6, 4, 5, 4]
    feature_names = ["S_Semantic", "Phrase", "Phase", "Semantic"]
    held_out_idxs = [4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24 ]
    
    create_datasets(
        audio_dir, text_vec_dir, gest_prop_dir, elan_dir,
        feature_names, feature_dims, 
        output_dir,
        held_out_idxs
    )
