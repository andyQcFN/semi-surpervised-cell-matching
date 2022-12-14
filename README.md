# A semi-supervised deep learning solution to cell registration in video data from calcium imaging studies

This repository is related to my thesis project about a deep learning approach to cell registration for calcium recordings.
* the “train” folder includes a pre-trained generator model that allows the generation of synthetic cell registration data. It also includes the script “CenterSurrNet.py” that allows you to train a center-surround network from scratch.

    * To generate a semi-synthetic dataset (you can adjust parameters such as the size of the dataset):
        ```
        python createDatasetOptimizedConv.py --dataset_size 10
        ```
    * To train the center-surround network from scratch use the following command (after a synthetic dataset is created):
        ```
        python CenterSurrNet.py
        ```


* the “test” folder includes a pre-trained center-surround network and a testing python script to perform cell registration on an example dataset. Note the input file formats to the python script. 

    * To run the example test:
    
        ```
        python 2streamCenterSurrRealNeuronMatching.py
        ```

Special thanks to Dr. Fung and Dr. Wong for supervising me throughout this project
