import os
import zipfile
import shutil
import math

def extract_zip(zip_file_path, extract_to_dir):
    """
    Extracts the contents of a ZIP file to a specified directory. If the specified directory already exists, it will be overwritten.

    Parameters:
        zip_file_path (str): Path to the ZIP file.
        extract_to_dir (str): Directory where contents will be extracted.

    Returns:
        None: Prints status messages during extraction.
    """
    if not os.path.exists(zip_file_path):
        print(f"Error: The file {zip_file_path} does not exist.")
        return
    
    if os.path.exists(os.path.join(extract_to_dir, 'svd')):
        shutil.rmtree(os.path.join(extract_to_dir, 'svd'))
        print(f"Deleted existing directory at {os.path.join(extract_to_dir, 'svd')}")

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_dir)
        print(f"Extracted {zip_file_path} to {extract_to_dir}")

    macosx_path = os.path.join(extract_to_dir, '__MACOSX')
    if os.path.exists(macosx_path) and os.path.isdir(macosx_path):
        shutil.rmtree(macosx_path)
        print(f"Removed _MACOSX directory at {macosx_path}")

def round_up_to_half(number):
    """
    Rounds a float up to the nearest multiple of 0.5.
    
    Parameters:
    number (float): The number to round.

    Returns:
    float: The number rounded up to the nearest 0.5.
    """
    return math.ceil(number * 2) / 2
