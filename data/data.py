from __future__ import print_function, division

import hashlib
import os
import json


def generate_file_md5(filename, blocksize=2**20):
    m = hashlib.md5()
    with open(filename, "rb") as f:
        while True:
            buf = f.read(blocksize)
            if not buf:
                break
            m.update(buf)
    return m.hexdigest()

def check_hashes(d):
    """
    Parameter
    -----------
    d : dictionary in which keys are file names and the items are hashes
        corresponding to the keys.
    
    Return
    --------
    check whether the has of each file key in 'd' matches the corresponding
    hash item.
    """

    all_good = True
    for k, v in d.items():
        digest = generate_file_md5(k)
        if v == digest:
            print("The file {0} has the correct hash.".format(k))
        else:
            print("ERROR: The file {0} has the WRONG hash!".format(k))
            all_good = False
    return all_good

def get_hash(directory):
    """
    Parameters
    -----------
    directory: the path of the directory you want to make a hash list for.
    Return:
    a list of hashes for each path in directory.
    """
    file_paths = []
    for path, subdirs, files in os.walk(directory):
        for name in files:
            file_paths.append(os.path.join(path, name))
    dictionary = {path: generate_file_md5(path) for path in file_paths}
    with open('sub001_hashes.json', 'w') as out:
        json.dump(dictionary, out)
    return dictionary
    

#get hashes for all files in all subdirectories of the decompressed 
#ds115_sub001 directory.
if __name__ == "__main__":
    file_hashes = get_hash('sub001')
    check_hashes(file_hashes)


