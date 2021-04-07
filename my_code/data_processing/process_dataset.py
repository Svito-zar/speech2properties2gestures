"""
This script does the preprocessing of the dataset specified in --proc_data_dir,
and stores the results in the same folder as .npy files.
It should be used before training, as described in the README.md file.
from my_code.data_processing.data_params import processing_argparser
 
@authors: Taras Kucherenko, Rajmund Nagy
"""

from my_code.data_processing.data_params import processing_argparser

import os
from os.path import join
from my_code.data_processing import tools
from my_code.data_processing import utils

import numpy as np
import h5py

def are_data_indices_matching(audio_fname, text_fname, motion_fname):
    """Check if the given files belong to the same data sample."""
    # The files have names like V05.hdf5
    return utils.get_file_name(audio_fname) == \
           utils.get_file_name(text_fname) == \
           utils.get_file_name(motion_fname)

def process_audio(audio_file):
    """Process selected features (Spectrogram or MFCC) of the given audio file at 25 Hz."""
    if args.feature_type == "MFCC":
        return tools.calculate_mfcc(audio_file, fps=25)

    elif args.feature_type == "Spectro":
        return tools.calculate_spectrogram(audio_file, fps=25)

    else:
        print(f"ERROR: Unknown audio feature type '{args.feature_type}'!")
        exit(-1)

def process_text(text_file, total_duration_ms):
    """
    Process the given text embeddings and upsample them 
    so that they're aligned with the audio frames.
    """
    # text_data has shape (n_words, 2 + n_features)
    text_data = h5py.File(text_file, mode='r').get("text")
    # the first two elements of the feature axis are word start/word end
    word_starts_ms = (text_data[:, 0] * 1000).round()
    word_ends_ms = (text_data[:, 1] * 1000).round()
    # and the rest is the word embedding vector
    word_embeddings = text_data[:, 2:]
    silence_embedding = np.array([-15] * 768) # BERT has 768-dim. features
    
    n_words = len(text_data)
    curr_word_id = 0
    elapsed_milliseconds = 0
    frame_duration_ms = (1 / 25) * 1000

    text_features = []
    
    
    # Process the silence before the first spoken word
    while elapsed_milliseconds < word_starts_ms[0]:
        word_extra_features = extract_word_extra_features(total_elapsed_ms = elapsed_milliseconds,
                                                          word_start_ms = 0,
                                                          word_end_ms = word_starts_ms[0])      
        
        word_feature_vector = np.concatenate([silence_embedding, word_extra_features])

        text_features.append(word_feature_vector)
        elapsed_milliseconds += frame_duration_ms

    # Process all the words in the recording
    while elapsed_milliseconds < total_duration_ms:
       
        # Process the word itself
        while elapsed_milliseconds < word_ends_ms[curr_word_id]:
            word_extra_features = extract_word_extra_features(total_elapsed_ms = elapsed_milliseconds,
                                                              word_start_ms = word_starts_ms[curr_word_id],
                                                              word_end_ms = word_ends_ms[curr_word_id])
            
            word_feature_vector = np.concatenate([word_embeddings[curr_word_id], word_extra_features])

            text_features.append(word_feature_vector)
            elapsed_milliseconds += frame_duration_ms
        
        if curr_word_id >= n_words-1:
            # At this point we have processed every word
            break

        # Process the silence before the next word
        while elapsed_milliseconds < word_starts_ms[curr_word_id + 1]:
            word_extra_features = extract_word_extra_features(total_elapsed_ms = elapsed_milliseconds,
                                                              word_start_ms = word_ends_ms[curr_word_id],
                                                              word_end_ms = word_starts_ms[curr_word_id + 1])      
            
            word_feature_vector = np.concatenate([silence_embedding, word_extra_features])

            text_features.append(word_feature_vector)
            elapsed_milliseconds += frame_duration_ms

        curr_word_id += 1
            
    # Process the silence after the last word
    while elapsed_milliseconds < total_duration_ms:
        word_extra_features = extract_word_extra_features(total_elapsed_ms = elapsed_milliseconds,
                                                          word_start_ms = word_ends_ms[-1],
                                                          word_end_ms = total_duration_ms)
        
        word_feature_vector = np.concatenate([silence_embedding, word_extra_features])
    
        text_features.append(word_feature_vector)
        elapsed_milliseconds += frame_duration_ms
    
    text_features = np.array(text_features)

    return text_features

def extract_word_extra_features(total_elapsed_ms, word_start_ms, word_end_ms):
    """Return extra features for the current text frame as a list.
    The four additional features are: 
        
        1) elapsed time since the beginning of the current word 
        2) remaining time from the current word
        3) the duration of the current word
        4) the progress as the ratio 'elapsed_time / duration'
    
    NOTE: The original Gesticulator also includes 
               5) the pronunciation speed of the current word (number of syllables per decisecond)
          But we skip this feature in our baseline.
      
    Args:
        total_elapsed_ms:    The elapsed time since the beginning of the entire recording
        word_start_ms:       The timestamp of the word's beginning
        word_end_ms:         The timestamp of the word's end
    
    Returns: 
        frame_extra_features:  A list that contains the 4 additional features.
    """
    word_elapsed_ms   = total_elapsed_ms - word_start_ms
    word_remaining_ms = word_end_ms - total_elapsed_ms
    word_duration_ms  = word_end_ms - word_start_ms
    word_progress     = word_elapsed_ms / word_duration_ms

    frame_extra_features = [
        word_elapsed_ms, 
        word_remaining_ms,
        word_duration_ms, 
        word_progress
    ]

    return frame_extra_features

def process_motion(motion_file):
    """Load the motion clips (in the exponential map format) in the given motion file."""
    motion_data = np.load(motion_file)['clips']

    return motion_data

def encode_as_vectors(audio_file, text_file, motion_file, dataset_split, add_context):
    """
    Load and process the given data files, and return them as numpy vectors.

    Args:
        audio_file:     Path to an audio file of format .mov.wav
        text_file:      Path to a processed text file of format .hdf5, containing word embeddings
        motion_file:    Path to a processed motion file of format .npz (originally a BVH)
        dataset_split:  The name of the dataset split to be encoded. Controls the sequence lengths 
                        when context is added.
        add_context:    Whether to add context in a rolling window manner.

    Returns:
        audio_vectors, text_vectors, motion_vectors:  
            Arrays of shape (n_windows, window_size, n_features), if add_context is True
                   or of shape (n_frames, n_features),            if add_context is False
    """
    print("\t-", utils.get_file_name(audio_file), end=" ")

    print("(processing audio...", end=" ", flush=True)
    audio_vectors = process_audio(audio_file)
    print(utils.tick_char, end="")
    
    print(" | text...", end=" ", flush=True)
    # TODO we currently upsample the text to 25 FPS, unlike the old 
    #      implementation which kept it at 10FPS -> memory issues?
    audio_duration_ms = (len(audio_vectors) / 25) * 1000
    text_vectors = process_text(text_file, total_duration_ms = audio_duration_ms)
    print(utils.tick_char, end="")

    assert len(audio_vectors) == len(text_vectors)

    print(" | motion...", end=" ", flush=True)
    motion_vectors  = process_motion(motion_file)
    print(utils.tick_char, end=")\n")
    
    audio_vectors, text_vectors, motion_vectors = \
        shorten_data_vectors(audio_vectors, text_vectors, motion_vectors)

    if add_context:
        audio_vectors, text_vectors, motion_vectors = \
            augment_with_context(audio_vectors, text_vectors, motion_vectors, dataset_split)
    
    return audio_vectors, text_vectors, motion_vectors

def shorten_data_vectors(audio_vectors, text_vectors, motion_vectors):
    """Trim the given vectors so that their length matches."""
    input_len = len(audio_vectors)
    output_len = len(motion_vectors)

    if input_len > output_len:
        audio_vectors = audio_vectors[:output_len]
        text_vectors = text_vectors[:output_len]
    else:
        motion_vectors = motion_vectors[:input_len]

    return audio_vectors, text_vectors, motion_vectors

def augment_with_context(audio_vectors, text_vectors, motion_vectors, dataset_split):
    """
    Process the given vectors in a rolling window manner using
    the parameters in 'args' (which is read from the command line).
    """
    if dataset_split == "train":
        sequence_length = args.sequence_length
    elif dataset_split == "dev":
        sequence_length = 5 * args.sequence_length 
    else:
        print("ERROR: Context augmentation is only enabled for the 'dev' and 'train' splits,")
        print(f"       but the current split is '{dataset_split}'!")
        exit(-1)

    assert len(audio_vectors) == len(text_vectors) == len(motion_vectors)
    assert args.past_context % 2 == 0
    assert args.future_context % 2 == 0
    hop = 40
    assert hop % 2 == 0

    start_ind = args.past_context
    n_reserved_inds = sequence_length + args.future_context
    stop_ind = len(audio_vectors) - n_reserved_inds
    
    def apply_context_windowing(input_array):
        windowed_array = np.array(
            [ input_array[i - args.past_context : i + n_reserved_inds]
              for i in range(start_ind, stop_ind, hop) ]
        )
        return windowed_array

    audio_vectors = apply_context_windowing(audio_vectors)
    text_vectors = apply_context_windowing(text_vectors)
    motion_vectors = apply_context_windowing(motion_vectors)

    return audio_vectors, text_vectors, motion_vectors

def process_dataset():
    """Go through each split in the dataset, process the files in them, and save the processed vectors."""
    # Data splits: train, dev, test
    for data_split in sorted(os.listdir(args.proc_data_dir)):
        data_split_dir = join(args.proc_data_dir, data_split)

        if os.path.isdir(data_split_dir):
            print(f"\nProcessing the '{data_split}' dataset...")
    
            input_dir    = join(data_split_dir, "inputs")
            motion_dir   = join(data_split_dir, "labels")

            audio_files  = [join(input_dir, file) for file in sorted(os.listdir(input_dir))
                            if file.endswith(".mov.wav")]

            text_files   = [join(input_dir, file) for file in sorted(os.listdir(input_dir))
                            if file.endswith(".hdf5")]

            motion_files = [join(motion_dir, file) for file in sorted(os.listdir(motion_dir))
                            if file.endswith(".npz")]
            
            if data_split in ["train", "dev"]:
                data_files = zip(audio_files, text_files, motion_files)
                process_and_save_dataset(data_files, args.proc_data_dir, data_split)
            
            if data_split in ["dev"]: #, "test"]:
                data_files = zip(audio_files, text_files, motion_files)
                process_and_save_data_as_sequences(data_files, data_split_dir, data_split)

def process_and_save_dataset(data_files, output_dir, data_split):
    """Process the given files and concatenate the results into one array per data modality."""
    audio_data = []
    text_data = []
    motion_data = []
    print("  Merging the following files for training:")

    for audio_file, text_file, motion_file in data_files:
        assert are_data_indices_matching(audio_file, text_file, motion_file)
        
        audio, text, motion = encode_as_vectors(
            audio_file, text_file, motion_file, data_split, add_context = True)
        
        audio_data.append(audio)
        text_data.append(text)
        motion_data.append(motion)
    
    audio_array  = np.concatenate(audio_data,  axis=0)
    text_array   = np.concatenate(text_data,   axis=0)
    motion_array = np.concatenate(motion_data, axis=0)
    
    print("  Merged dataset shapes:\n    audio:  {}\n    text:   {}\n    motion: {}".format(
        audio_array.shape, text_array.shape, motion_array.shape))

    np.save(join(output_dir, f'X_{data_split}'), audio_array)
    np.save(join(output_dir, f'T_{data_split}'), text_array)
    np.save(join(output_dir, f'Y_{data_split}'), motion_array)

def process_and_save_data_as_sequences(data_files, output_dir, data_split):
    """Process the given files and save them separately."""
    print("  Saving the following sequences for evaluation:")

    for audio_file, text_file, motion_file in data_files:
        assert are_data_indices_matching(audio_file, text_file, motion_file)

        audio, text, motion = encode_as_vectors(
            audio_file, text_file, motion_file, data_split, add_context = False)

        filename = utils.get_file_name(audio_file)
        np.save(join(output_dir, "sequences", 'X_' + filename.lstrip('V') + '.npy'), audio)
        np.save(join(output_dir, "sequences", 'T_' + filename.lstrip('V') + '.npy'), text)
        np.save(join(output_dir, "sequences", 'Y_' + filename.lstrip('V') + '.npy'), motion)

if __name__ == "__main__":
    args = processing_argparser.parse_args()

    process_dataset()
  
