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

def split_annotation_into_labels(annotation):
    """
    Split the given annotation string into a list of labels.
    """
    # NOTE: re.split() always returns a list, so we use the flatten function to avoid 
    #       nested lists, e.g. flatten([['deictic'], ['iconic']]) = ['deictic', 'iconic']
    flatten = itertools.chain.from_iterable
    
    # Split on all separators that are found in the data
    labels = [annotation]
    for sep in ['-', '/', ',', '\n']:
        labels = flatten([re.split(sep, part) for part in labels])
        
    # Remove leading/trailing whitespace
    labels = [part.strip() for part in labels]
    
    return labels
    
def extract_labels(annotation_str):
    """
    Split the given annotation string into individual labels, while fixing known typos.
    """
    annotation_str = annotation_str.lower().strip()
    if annotation_str == "":
        return []
        
    # Remove special characters
    special_chars = "!?@#%^&*_"
    for char in special_chars:
        annotation_str = annotation_str.replace(char, "")
        
    if annotation_str == "entity relative position":
        annotation_str = "entity-relative position"
    elif annotation_str == "relative positionm amount":
        annotation_str = "relative position-amount"
    
    # Split combined labels e.g. iconic-deictic -> [iconic, deictic]
    labels = split_annotation_into_labels(annotation_str)    
    labels = [fix_typos(label) for label in labels]

    return labels
