import itertools
import re

def fix_typos(label):
    """
    Fix known typos in the given individual (not compound) label.
    """
    typo_dict = {
        "relatie position" : "relative position",
        "relatve position" : "relative position",
        "relative posiiton": "relative position",
        "reltive position" : "relative position",
        "saggital" : "sagittal",
        "shape38"  : "shape",
        "entities" : "entity",
        "enitity"  : "entity",
        "enity"    : "entity"
    }

    return typo_dict[label] if label in typo_dict else label

def split_label(label):
    """
    Split combined labels into a list of single labels (e.g. 'deictic-beat' to ['deictic', 'beat']).
    """
    flatten = itertools.chain.from_iterable
    
    # Split on all separators (sequentially) that are found in the data
    label_parts = [label]
    for sep in ['-', '/', ',', '\n']:
        # NOTE: 'flatten' squashes the lists that re.split() returns
        label_parts = flatten([re.split(sep, part) for part in label_parts])
        
    
    # Remove leading/trailing whitespace
    label_parts = [part.strip() for part in label_parts]
    
    return label_parts
    
def clean_and_split_label(label):
    """
    Fix typos and return the the given (potentially compound) label
    as a list of individual labels.
    """
    label = label.lower().strip()

    # Remove special characters
    special_chars = "!?@#%^&*_"
    for char in special_chars:
        label = label.replace(char, "")
        
    if label == "entity relative position":
        label = "entity-relative position"
    elif label == "relative positionm amount":
        label = "relative position-amount"
    
    # Split combined labels e.g. iconic-deictic -> [iconic, deictic]
    label_parts = split_label(label)    
    label_parts = [fix_typos(part) for part in label_parts]

    return label_parts
