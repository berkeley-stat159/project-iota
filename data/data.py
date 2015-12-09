from __future__ import print_function, division

import json
from get_hashes import check_hashes


# check ds115_sub001 data.
if __name__ == "__main__":
    with open('sub001_hashes.json', 'w') as out:
        d = json.load(out)
    check_hashes(d)
