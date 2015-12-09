import json

from get_hashes import get_hash



#get hashes for all files in all subdirectories of the decompressed 
#sub001 directory.
if __name__ == "__main__":
    file_hashes = get_hash('sub001')
    with open('sub001_hashes.json', 'w') as out:
        json.dump(file_hashes,out)

