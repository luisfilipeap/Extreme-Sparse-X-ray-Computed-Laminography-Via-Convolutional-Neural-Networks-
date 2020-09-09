# Extreme Sparse X-ray Computed Laminography Via Convolutional Neural Networks
Code used to run the experiments of the paper published on ICTAI 2020

A. Tomography Folder

    Execute the following:

        1. run object_scan.py to get sparse-view inline X-ray CT reconstructions            
    
B. Network Folder

    Execute the following:

        1. run build_npy.py to obtain the correct format of output files
        2. run data_utils.py to create csv files for training, validation, and test
            (you can skip this step and use the csv files I generated)
        3. run train.py
        4. execute generalization.py to get the results for the images in the test set
