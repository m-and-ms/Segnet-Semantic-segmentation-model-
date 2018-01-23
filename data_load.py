from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np
import itertools

from helper import *
import os

DataPath = '/home/bora3i/Downloads/may_segnet/CamVid/'
data_shape = 360*480

#types is the type of data as [train,test,validation]
def load_data(types):
    data = []
    label = []
    with open(DataPath + types +'.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
    for i in range(len(txt)):
        data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + txt[i][0][7:])),2))
        label.append(one_hot_it(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[:,:,0]))
        print('.',end='')
    return np.array(data), np.array(label)



train_data, train_label = load_data("train")
train_label = np.reshape(train_label,(367,data_shape,12))

test_data, test_label = load_data("test")
test_label = np.reshape(test_label,(233,data_shape,12))

val_data, val_label = load_data("val")
val_label = np.reshape(val_label,(101,data_shape,12))


np.save("data/train_data", train_data)
np.save("data/train_label", train_label)

np.save("data/test_data", test_data)
np.save("data/test_label", test_label)

np.save("data/val_data", val_data)
np.save("data/val_label", val_label)


