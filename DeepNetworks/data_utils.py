# title                 :data_utils.py
# description           :It creates a training-validation-test split and computes the mean gray value in the images of the training set
# author                :Dr. Luis Filipe Alves Pereira (luis.filipe-pereira@ufape.edu.br)
# date                  :2019-05-16
# version               :1.0
# notes                 :Please let me know if you find any problem in this code
# python_version        :3.6
# numpy_version         :1.16.3
# scipy_version         :1.2.1
# matplotlib_version    :3.0.3
# pilow_version         :6.0.0
# pandas_version        :0.24.2
# pytorch_version       :1.1.0
# ==============================================================================


import os
import numpy as np
import pandas as pd
from math import floor

import random
from imageio import imread


"""
Parameters to split data into groups for training, validation, and testing  

src_img:            directory containing the set of low quality or high quality images 
split_proportion:   proportion of data into the training, validation, and testing groups respectively
"""


src_img = "D:\\Datasets\\demo_data_plates_64_-1\\input\\"
split_proportion = [.2, .1, .1]




def get_train_val_test(src_data, proportion):

    scans = os.listdir(src_data)
    random.shuffle(scans)

    t1 = floor(len(scans) * proportion[0])
    t2 = t1 + floor(len(scans) * proportion[1])
    t3 = t2 + floor(len(scans) * proportion[2])
    training = scans[0:t1]
    validation = scans[t1:t2]
    test = scans[t2:t3]

    return training, validation, test

def create_csv_files(src_data, proportion):

    if not os.path.isfile('train_ictai.csv') and not os.path.isfile('validation_ictai.csv') and not os.path.isfile('train_ictai.csv'):
        train_file = open('train_ictai.csv','w')
        val_file = open('validation_ictai.csv','w')
        test_file = open('test_ictai.csv','w')

        train_set, val_set, test_set= get_train_val_test(src_data, proportion)

        for z in train_set:
            train_file.write(z + ',' + z+ '.npy' + '\n')

        for z in val_set:
            val_file.write(z + ',' + z+ '.npy' + '\n')

        for z in test_set:
            test_file.write(z + ',' + z+ '.npy' + '\n')

        train_file.close()
        val_file.close()
        test_file.close()
    else:
        print('Data already splitted into training, validation, and testing')

def data_mean_value(csv, dir):
    data = pd.read_csv(csv)
    r, c = data.shape
    slices_per_scan = 30
    values = np.zeros((r*slices_per_scan,1))
    idx = 0
    for i, row in data.iterrows():
        for file in os.listdir(dir+row[0]):
            img = imread(dir+row[0]+'\\'+file, pilmode="F")
            values[idx,:] = np.mean(img, axis=(0,1))
            idx += 1
    #print(values)
    return np.mean(values,axis=0)



if __name__ == "__main__":
    create_csv_files(src_img, split_proportion)
    #print(data_mean_value("test.csv", src_img))
    #print(data_mean_value("train.csv", src_img))
