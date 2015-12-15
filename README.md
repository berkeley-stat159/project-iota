i# project-iota
STAT 159/259: Reproducible and Collaborative Statistical Data Science

Fall 2015 UC Berkeley


[![Join the chat at https://gitter.im/Jay4869/project-iota](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/Jay4869/project-iota?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/berkeley-stat159/project-iota.svg?branch=master)](https://travis-ci.org/berkeley-stat159/project-iota?branch=master)
[![Coverage Status](https://coveralls.io/repos/berkeley-stat159/project-iota/badge.svg?branch=master)](https://coveralls.io/r/berkeley-stat159/project-iota?branch=master)

This repository contains reproducible analysis based on dataset used in [Working Memory in Healthy and Schizophrenic Individuals] (https://openfmri.org/dataset/ds000115).

## Installation
1. Clone our project repository: `https://github.com/berkeley-stat159/project-iota`
2. Install python modules with pip: `pip install -r requirements.txt`

## General steps:
#### Download Dataset
- `make dataset`: Download the 4.2GB `ds115_sub001-005.tgz` and `nipy.bic.berkeley.edu`, then automatically unzip `sub001`, the only subject data that we worked with.

#### Generate Convolution
- `make convo`: Convolve study conditions with hemodynamic function(HDF). Included are the regular block design convolutions as well as a higher resolution convolutions for event-related convolutions.

#### Modeling & Analysis
- `make modeling`: Run all the regressions and related plots mentioned in our report. 

#### Hypothesis Testing
- `make testing`: Run all the hypothesis tests and validations of our mode.

#### Report
- `make report`: Compile our final report with analysis results.


Thanks to [Jarrod Millman](https://github.com/jarrodmillman), [Matthew Brett](https://github.com/matthew-brett), [Ross Barnowski] (https://github.com/rossbar) and [J-B Poline] (https://github.com/jbpoline) for their instructions throughout the semester.


## Contributors

- Jie Li ([`Jay4869`](https://github.com/Jay4869))
- Zeyu Li ([`lizeyuyuz`](https://github.com/lizeyuyuz))
- Qingyuan Zhang ([`amandazhang`](https://github.com/amandazhang))
- Yun Chuan ([`ay2456`](https://github.com/ay2456))
