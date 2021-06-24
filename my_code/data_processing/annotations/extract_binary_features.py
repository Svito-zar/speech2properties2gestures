from tqdm import tqdm
from os import path
from os.path import join
import pympi
import numpy as np
import os
import pickle
import h5py
from pprint import pprint
from my_code.data_processing.annotations.utils import extract_labels

def open_elan_tier_for_property(elan_object, property_name):
    """
    Return the ELAN tier for the given gesture property.
    """
    if property_name not in elan_object.tiers:
        raise KeyError(f"The '{property_name}' annotations are missing from the current file!")
    
    tier = elan_object.tiers[property_name]

    return tier[0] if len(tier[0]) > 0 else tier[1]

def open_or_create_label_dict(annotation_folder, dict_file, properties_to_consider):
    """
    Open or create a dictionary that maps property names to their index-label pairs.

    Args:
        annotation_folder:  path to the annotation folder
        dict_file:  the save path of the dictionary
        properties_to_consider: the properties to use when the dict is created
    Returns:
        The dictionary that maps properties to their labels.
    """
    if path.isfile(dict_file):
        with open(dict_file, 'rb') as handle:
            print(f"Opening label dictionary: '{dict_file}'.")
            return pickle.load(handle)

    print(f"Creating label dictionary: '{dict_file}'.")
    label_dict = {}

    progress_bar = tqdm(properties_to_consider)
    for property_name in progress_bar:
        progress_bar.set_description(f"Parsing [{property_name}] annotations]")
        possible_labels = set()

        # go through all the files in the dataset
        for filename in sorted(os.listdir(annotation_folder)):
            if not filename.endswith("eaf"):
                continue

            annotation_path = join(annotation_folder, filename)
            elan_object = pympi.Elan.Eaf(file_path=annotation_path)

            try:
                elan_tier = open_elan_tier_for_property(elan_object, property_name)
            except KeyError as missing_property_warning:
                print(f"WARNING: {missing_property_warning}")
                continue

            for annotation_entry in elan_tier.values():
                assert len(annotation_entry) == 4
                st_t, end_t, annotation_str, _ = annotation_entry
                
                # Sometimes they are messed up
                if annotation_str is None:
                    st_t, annotation_str, _, _ = annotation_entry

                labels = extract_labels(annotation_str)
                for label in labels:
                    possible_labels.add(label)
        
        # Explicitly store the label indices alongside the values, in alphabetic order
        label_dict[property_name] = {i: val for i, val in enumerate(sorted(possible_labels))}
        print(f"\n-----> '{property_name}' labels:", list(label_dict[property_name].values()), end="\n\n")
    # Save the dictionary
    f = open(dict_file, "wb")
    pickle.dump(label_dict, f)
    f.close()

    return label_dict

def encode_selected_properties(elan_object, output_hdf5_object, properties_to_consider):
    """
    Encode the selected gesture properties as binary feature vectors and save them
    into the given hdf5 object.

    Args:
        elan_object:  an ELAN object containing the gesture property annotations
        output_hdf5_object:  the hdf5 object where the feature vectors will be saved
        properties_to_consider:  the names of the properties that will be saved

    Returns:
        Nothing, but the feature vectors are saved into 'output_hdf5_object'.
    """
    timeslots = elan_object.timeslots

    for property_name in properties_to_consider:
        try:
            # The ELAN tier contains the annotations and their timestamps
            curr_elan_tier = open_elan_tier_for_property(elan_object, property_name)
        except KeyError as missing_property_warning:
            print(f"WARNING: {missing_property_warning}")
            # TODO(RN): this used to be a break, check with Taras if it was intended
            continue

        property_features = []
        label_dim_to_name = PROPERTY_DIM_TO_LABEL[property_name]

        for annotation_entry in curr_elan_tier.values():        
            assert len(annotation_entry) == 4
            start_t, end_t, annotation_str, _ = annotation_entry                  

            # Skip annotations that have missing timestamps
            if timeslots[start_t] is None or timeslots[end_t] is None:
                continue
            
            # Skip missing annotations
            if annotation_str == "":
                continue

            present_labels = extract_labels(annotation_str)
            feature_vector = [0 for _ in range(len(label_dim_to_name))]
            
            for label_dim, label_name in label_dim_to_name.items():
                if label_name.lower() in present_labels:
                    feature_vector[label_dim] = 1

            time_n_feat = [timeslots[start_t] / 1000] + [timeslots[end_t] / 1000] + feature_vector

            property_features.append(np.array(time_n_feat))

        output_hdf5_object.create_dataset(
            name = property_name, 
            data = np.array(property_features)
        )

def encode_phrase_practice(elan_object, output_hdf5_object):
    """
    Encode the Phrase and the corresponding Practice labels as binary feature
    vectors, and save them into 'output_hdf5_object'.
    
    Args:
        elan_object:  an ELAN object containing the gesture property annotations
        output_hdf5_object:  the hdf5 object where the feature vectors will be saved
    
    Returns:
        Nothing, but the feature vectors are saved into 'output_hdf5_object'.
    """
    timeslots = elan_object.timeslots
 
    for hand in ["Left", "Right"]:
        phrase_property = f"R.G.{hand}.Phrase"
        practice_property = f"R.G.{hand}.Practice"

        phrase_dim_to_label = PROPERTY_DIM_TO_LABEL[phrase_property]
        practice_dim_to_label = PROPERTY_DIM_TO_LABEL[practice_property]
        
        try:
            phrase_tier = open_elan_tier_for_property(elan_object, phrase_property)
            practice_tier = open_elan_tier_for_property(elan_object, practice_property)
        except KeyError as missing_property_warning:
            print(f"WARNING: {missing_property_warning}")
            continue # TODO(RN): this used to be break, why?
        
        feature_vectors = []

        for practice_annotation_entry in practice_tier.values():        
            # 1. Extract practice property labels
            ges_key, practice_annotation, _, _ = practice_annotation_entry

            # Skip missing annotations
            if practice_annotation.strip() == "":
                continue

            practice_feature_vector = [0 for _ in range(len(practice_dim_to_label))]
            present_practice_labels = extract_labels(practice_annotation)
            
            for label_dim, label_name in practice_dim_to_label.items():
                if label_name.lower() in present_practice_labels:
                    practice_feature_vector[label_dim] = 1

            # 2. Extract the corresponding parent Phrase info
            st_t, end_t, phrase_annotation, _  = phrase_tier[ges_key]

            if timeslots[st_t] is None or timeslots[end_t] is None:
                continue

            # Skip missing annotations
            if phrase_annotation.strip() == "":
                continue

            present_phrase_labels = extract_labels(phrase_annotation)
            phrase_feature_vector = [0 for _ in range(len(phrase_dim_to_label))]

            for label_dim, label_name in phrase_dim_to_label.items():
                if label_name.lower() in present_phrase_labels:
                    phrase_feature_vector[label_dim] = 1

            start_time = timeslots[st_t] / 1000
            end_time = timeslots[end_t] / 1000
            
            time_n_feat = [start_time, end_time] + phrase_feature_vector + practice_feature_vector
            feature_vectors.append(np.array(time_n_feat))

        feature_vectors = np.array(feature_vectors)

        dataset_name = f"gesture_phrase_n_practice_{hand}"
        output_hdf5_object.create_dataset(dataset_name, data=feature_vectors)

def encode_semantic_labels(elan_object, output_hdf5_object):
    """
    Encode the Semantic gesture properties as binary feature vectors and save them
    into the given hdf5 object.
    
    Args:
        elan_object:  an ELAN object containing the gesture property annotations
        output_hdf5_object:  the hdf5 object where the feature vectors will be saved
    
    Returns:
        Nothing, but the feature vectors are saved into 'output_hdf5_object'.
    """
    timeslots = elan_object.timeslots
 
    for hand in ["Left", "Right"]:
        phrase_property = f"R.G.{hand}.Phrase"
        semantic_property = f"R.G.{hand} Semantic"
        semantic_dim_to_label = PROPERTY_DIM_TO_LABEL[semantic_property]
        
        try:
            phrase_tier = open_elan_tier_for_property(elan_object, phrase_property)
            semantic_tier = open_elan_tier_for_property(elan_object, semantic_property)
        except KeyError as missing_property_warning:
            print(f"WARNING: {missing_property_warning}")
            continue # TODO(RN): this used to be break, why?
        
        feature_vectors = []

        for semantic_annotation_entry in semantic_tier.values():        
            # 1. Extract semantic property labels
            sem_st_t, sem_end_t, semantic_annotation, _ = semantic_annotation_entry

            # Skip missing annotations
            if semantic_annotation.strip() == "":
                continue

            semantic_feature_vector = [0 for _ in range(len(semantic_dim_to_label))]
            present_semantic_labels = extract_labels(semantic_annotation)
            
            for label_dim, label_name in semantic_dim_to_label.items():
                if label_name.lower() in present_semantic_labels:
                    semantic_feature_vector[label_dim] = 1

            # 2. Find the timing of the corresponding Phrase
            for phrase_annotation_entry in phrase_tier.values():
                phr_st_t, phr_end_t, phrase_annotation, _ = phrase_annotation_entry
                if timeslots[phr_st_t] is None or timeslots[phr_end_t] is None:
                    continue
                if timeslots[phr_end_t] >= timeslots[sem_st_t]:
                    break

            # Save the Semantic features with the timing of the whole Phrase
            time_n_feat = [timeslots[phr_st_t] / 1000] + [timeslots[phr_end_t] / 1000] + semantic_feature_vector
            feature_vectors.append(np.array(time_n_feat))

        output_hdf5_object.create_dataset(
            name = semantic_property, 
            data = np.array(feature_vectors)
        )

def create_hdf5_file(annotation_filename):
    """
    Create the output hdf5 object based on the ELAN filename.
    """
    file_idx = annotation_filename[:2]
    hdf5_file_name = join("feat/gesture_properties/", f"{file_idx}_feat.hdf5")
    
    assert os.path.isfile(hdf5_file_name) == False
    
    return h5py.File(name=hdf5_file_name, mode='w')

if __name__ == "__main__":
    all_properties = [
        'R.G.Left Semantic', 'R.G.Right Semantic',
        'R.G.Left.Phase',    'R.G.Right.Phase',
        'R.G.Left.Phrase',   'R.G.Right.Phrase',
        'R.G.Left.Practice', 'R.G.Right.Practice',
        'R.Movement_relative_to_other_Hand',
        'R.S.Pos' ,
        'R.S.Semantic Feature'
    ]
    
    annotation_folder = "/home/work/Desktop/repositories/probabilistic-gesticulator/dataset/All_the_transcripts/"

    dict_file = "dict.pkl"
    # TODO(RN) find a better name
    PROPERTY_DIM_TO_LABEL = open_or_create_label_dict(annotation_folder, dict_file, all_properties)
    pprint(PROPERTY_DIM_TO_LABEL)

    progress_bar = tqdm(sorted(os.listdir(annotation_folder)))
    for filename in progress_bar:
        if not filename.endswith(".eaf"):
            continue
        progress_bar.set_description(filename)

        annotation_file = join(annotation_folder, filename)
        elan_object = pympi.Elan.Eaf(file_path=annotation_file)
        hdf5_dataset = create_hdf5_file(filename)
        
        properties = [
            "R.G.Left.Phase",  "R.G.Right.Phase",
            "R.G.Left.Phrase", "R.G.Right.Phrase",
            "R.S.Semantic Feature"
        ]   

        encode_selected_properties(elan_object, hdf5_dataset, properties)
        encode_phrase_practice(elan_object, hdf5_dataset)
        encode_semantic_labels(elan_object, hdf5_dataset)
