# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 08:55:41 2020

@author: dujing1107
"""

import csv
import os          
import numpy as np       
import math
from sklearn.preprocessing import MinMaxScaler
import pickle

# Paths for results from dataset
main_path = "E:/sisFall/dataset/"
paths = [main_path + "/D01",main_path + "/D02", main_path + "/F01", main_path + "/F02"]

# Variable and array declarations for applying RNN
sample_num = 315                   # Setting number of desired samples
time_steps = 2500                   # Setting number of desired time steps
div_steps = 100
types = 4                          # Setting number of types you're processing
feat_num = 1                       # Setting number of desired features
RNNx = [[] for _ in range(types)]      # Array for RNN samples
RNNy = [[] for _ in range(types)]
RNNz = [[] for _ in range(types)]
vectorsumBF_RNN = [[] for _ in range(types)]  # Array for RNN vectorsums before filtering
deviation00 = [[] for _ in range(types)]
deviation01 = [[] for _ in range(types)]
train_samplesBF_RNN = []           # Array for all final RNN samples before filtering
train_labelsBF_RNN = []            # Array for all labels of 315 samples before filtering
scaled_train_samplesBF_RNN = []


# Function to calculate horizontal plane vectorsum feature

# C2
def vectorsum (Acc_x, Acc_z):
    x = []
    z = []
    print("Preparing vectorsum feature...")
    x = [i*i for i in Acc_x]
    z = [i*i for i in Acc_z]
    sum = [a+b for a,b in zip(x,z)]
    vectorsum_res = [math.sqrt(i) for i in sum]
    return vectorsum_res

# C8
def deviation0 (Acc_x, Acc_z):
    print("Preparing Standard deviation feature...")
    s = [0 for _ in range(100)]
    for i in range(100):
        s[i] = math.sqrt(np.var(Acc_z[i:i+25]) + np.var(Acc_x[i:i+25]))
    print("Preparing Standard deviation feature...")
    return s

# C8
def deviation1 (Acc_x, Acc_y,Acc_z):
    print("Preparing Standard deviation feature...")
    s = [0 for _ in range(100)]
    for i in range(100):
        s[i] = math.sqrt(np.var(Acc_x[i:i+25]) + np.var(Acc_y[i:i+25])+ np.var(Acc_y[i:i+25]))
    print("Preparing Standard deviation feature...")
    return s


feature = deviation01
func = deviation1

# Dividing data into time slices
for path in paths:
    j = paths.index(path)
    k = 0
    for file in os.scandir(path):
        RNNx[j].append([])
        RNNy[j].append([])
        RNNz[j].append([])
        with open(file, 'r') as RNNcsv:
            lines = RNNcsv.readlines()
            # Clearing white spaces
        with open(file, 'w') as RNNcsv:
            lines = filter(lambda x: x.strip(), lines)
            print("Clearing white spaces from files...")
            RNNcsv.writelines(lines)
            # Appending ADL x-axis & y-axis lists
        with open(file, 'rt') as RNNcsv:
            RNNdata = (RNNcsv.readlines()[0:time_steps])
            RNNsamples = csv.reader(RNNdata, delimiter=',')
            for i in RNNsamples:
                print("Importing accelerometer readings...")
                RNNx[j][k].append(int(i[0])* 32.0 / 8192.0)
                RNNy[j][k].append(int(i[1])* 32.0 / 8192.0)
                RNNz[j][k].append(int(i[2])* 32.0 / 8192.0)
        k = k+1

for i in range(len(RNNx)):
    for j in range(len(RNNx[i])):
        feature[i].append(func(RNNx[i][j], RNNy[i][j],RNNz[i][j]))


for i in range(len(feature)):
    for j in range(len(feature[i])):
        print("Appending RNN label and feature in array...")
        train_labelsBF_RNN.append(i)
        train_samplesBF_RNN.append(feature[i][j])

# Placing training samples and labels into numpy arays
train_samplesBF_RNN = np.array(train_samplesBF_RNN)
train_labelsBF_RNN = np.array(train_labelsBF_RNN)

# Scaling & reshaping data from 2D -> 3D
print("Scaling data samples...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_samplesBF_RNN = scaler.fit_transform((train_samplesBF_RNN).reshape(-1, 1))
scaled_train_samplesBF_RNN = scaled_train_samplesBF_RNN.reshape(sample_num, div_steps, feat_num)

# Saving data
print("Saving data samples before filtering...")
pickle_out = open("Samples_RNN.pickle", "wb")
pickle_out.truncate(0)
pickle.dump(scaled_train_samplesBF_RNN, pickle_out)
pickle_out.close()

pickle_out = open("Labels_RNN.pickle", "wb")
pickle_out.truncate(0)
pickle.dump(train_labelsBF_RNN, pickle_out)
pickle_out.close()