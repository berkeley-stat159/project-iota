from __future__ import print_function, division

import json
import sys
from . import get_hashes
from get_hashes import check_hashes


# check ds115_sub001 data.
if __name__ == "__main__":
    with open('sub001_hashes.json', 'r') as out:
        d = json.load(out)
    check_hashes(d)
