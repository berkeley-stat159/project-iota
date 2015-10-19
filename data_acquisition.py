"""
This module contains functions for downloading and verifying data from
the internet.
"""
import urllib2
import os.path
import nibabel
import json
from pprint import pprint

def download_data(url):
    """
    Download and save data from a url.

    Parameters
    ----------
    url : string
        The full url pointing to the data to be downloaded

    Returns
    -------
    data : str
         Return the data that results from reading the downloaded info, e.g.
         f.read() where f is the handle to the data buffer.

    Hint
    ----
    Consider the urllib2 or wget python modules
    """	
    return urllib2.urlopen(url).read()

def save_data(data, output_filename):
    """
    Save data to file.

    If the file already exists, the function will not save the data and return
    1

    Parameters
    ----------
    data : str
         String containing the data you wish to write out to a file

    output_filename : str
         Path (full or relative) to the file you will save the data into.

    Returns
    -------
    out : int
        Return 0 if the data was saved successfully. Return 1 if the file 
        already exists.

    Hint
    ----
    Check out the os module for determining whether a file exists already.
    """
    if os.path.isfile(output_filename):
        return 1
    else:    
        f = open(output_filename, 'w')
	for i in data:
	    f.write(i)
        f.close()        
     return 0 

def verify_data(data, known_checksum):
    """
    This function verifies the data by comparing it to the known sha1 hash for
    the given data.

    Parameters
    ----------
    data : str
         The data to be verified

    known_checksum : str
         The sha1 hash with which to compare the result of hashing data

    Returns
    -------
    out : bool
        Return True if the hashes match, False otherwise

    Hint
    ----
    Check out the hashlib module
    """
    return hashlib.sha1(data).hexdigest() == know_checksum
        
def load_parsed_data(fname):
    """
    Load fmri data in .nii and parse into a numpy array

    Parameters
    ----------
    fname : str
        Path to an .nii file containing fmri data
    
    Returns
    -------
    img_ary : numpy.core.memmap.memmap
        Parsed data will be stored in a numpy container

    Hint
    ----
    Use nibabel
    """
    img = nib.load(fname)
    data = img.get_data()
    return data

def main(data.json):
    """
    This function will be run if data_acquisition.py is called as a script. It
    is intended to be used with the %run method in ipython to initialize a
    session for data analysis

    This function should load a filename, url, and verified checksum from a 
    json archive. It will then check if a file with the given name already 
    exists in ../data. If not, download the data from the given url, save it
    to ../data/<filename>. Then verify the data. If the data verification
    passes, the data should be parsed into a useful numpy format.

    Parameters
    ----------
    data.json : str
        Path to a json file containing at least 'name', 'url', and 'sha1'
        fields

    Returns
    -------
    data : numpy.core.memmap.memmap
        The fMRI data in a numpy memmap representation, ready for analysis

    Hint
    ----
    Use the functions you've implemented above
    """
    with open('example_data.json') as data_file:
        data = json.load(data_file)
    pprint(data)
        
    url = str(data['url'])
    name = str(data['name'])
    sha1 = str(data['sha1'])
    fname = '../data/' + name

    if not os.path.isfile(fname):
        d = download(url) 
	save_data(d,fname)
	if verify_data(d, sha1):
	    bio_data = load_parsed_data(fname) 	

    return bio_data

### Add lines here that guarantees main() is run with example_data.json when
### called as a script
