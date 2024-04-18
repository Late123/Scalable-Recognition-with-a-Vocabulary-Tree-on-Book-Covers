import os
from PIL import Image
import hashlib

def file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def find_duplicates(folder):
    hashes = {}
    duplicates = []

    filecount = 0
    # Loop through all files in the directory
    for subdir, dirs, files in os.walk(folder):
        for filename in files:
            filecount += 1
            if filecount % 1000 == 0:
                print(f"Processed {filecount} files")
            if filename.endswith(('png', 'jpg', 'jpeg', 'gif')):
                file_path = os.path.join(subdir, filename)
                # Get the hash of the file
                img_hash = file_hash(file_path)
                
                if img_hash in hashes:
                    duplicates.append((file_path, hashes[img_hash]))
                else:
                    hashes[img_hash] = file_path

    return duplicates

def delete_files(file_list):

    """ Delete files and confirm deletion """
    for file in file_list:
        os.remove(file)
        print(f"Deleted {file}")

# Usage
folder_path = '..\\data\\train'
duplicates = find_duplicates(folder_path)
print(len(duplicates))

files_to_delete = [file for file, orig in duplicates]
delete_files(files_to_delete)