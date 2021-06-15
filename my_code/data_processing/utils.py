from pathlib import Path

tick_char = u'\u2713'

def get_file_name(file_path):
    """Get the file name from the path and return it without the extension."""
    return Path(file_path).name.split('.')[0]

def get_file_extension(file_path):
    """Get the file name from the path and return its extension."""
    file_name_with_extension = Path(file_path).name
    file_extension = ".".join(file_name_with_extension.split('.')[1:])
    
    return file_extension

def shorten(arr1, arr2):
    """Trim the one of the given arrays so that they have the same length."""
    min_len = min(len(arr1), len(arr2))

    return arr1[:min_len], arr2[:min_len]