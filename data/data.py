from __future__ import print_function, division

import hashlib
import os
import json


d = {'./sub001/BOLD/task001_run001/filtered_func_data_mni.nii.gz': '8e47b6000460e3b266e34c871e6fc00e','./sub001/BOLD/task003_run001/filtered_func_data_mni.nii.gz': '207fd1598d4785f7b217a3a3a4769430','./sub001/BOLD/task002_run001/filtered_func_data_mni.nii.gz': '4557a5218a224c48ea67ca8aa92b794d'}

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

    

#get hashes for all files in all subdirectories of the decompressed 
#ds115_sub001 directory.
if __name__ == "__main__":
    check_hashes(d)


