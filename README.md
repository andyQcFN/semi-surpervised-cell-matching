# A semi-supervised deep learning solution to cell registration in video data from calcium imaging studies

This repository is related to my thesis project about a deep learning approach to cell registration for calcium recordings.
* the “train” folder includes a pre-trained generator model that allows the generation of synthetic cell registration data. It also includes the script “CenterSurr.py” that allows you to train a center-surround network from scratch.

* the “test” folder includes a pre-trained center-surround network and a testing python script to perform cell registration on an example dataset. Note the input file formats to the python script. 
