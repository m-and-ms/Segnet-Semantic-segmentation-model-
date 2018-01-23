from __future__ import absolute_import
from __future__ import print_function
import os


import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization


from keras import backend as K

import cv2
import numpy as np
import json
np.random.seed(7)
#deconvilution and uosamoling are for increase in deimensinality 
#deconvilution can be convilution transpose in tensor flow 
#up sampling can be done by unpooling where the feature map produced will be sparse so we will need to re-convolve the output feature map to change sparse into dense 
#if we increased the dimensinalty by other methods like (deconvilution)or named (convilution transpose)it could do the job
#Their is also another methods to increase dimensinality like bilinear interpolation in tensorflow (beilinear risize )
iwidth = 480
iheight = 360
classn = 12
pz=2
kernel = 3
pad = 1

#entering [360*480]
#the convilution is done by [3*3]kernel in encdoing and 64 kernels 
#input image is 3 channel image RGB thats why kernel size is [3*3] 

def modelbuild():
    
    may_segnet = models.Sequential()
    may_segnet.add(Layer(input_shape=(3, 360, 480)))
    may_segnet.add(Convolution2D(64, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(Convolution2D(64, kernel, kernel, border_mode='same'))
     
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(MaxPooling2D(pool_size=(pz, pz)))

    #after first max pool [180*240]
    may_segnet.add(Convolution2D(128, kernel, kernel, border_mode='same'))

    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(Convolution2D(128, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(MaxPooling2D(pool_size=(pz, pz)))
    #after sec macpool [90*120
    may_segnet.add(Convolution2D(256, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(Convolution2D(256, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(Convolution2D(256, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    #after third max pool [45*60]
    may_segnet.add(MaxPooling2D(pool_size=(pz, pz)))
    may_segnet.add(Convolution2D(512, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(Convolution2D(512, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(Convolution2D(512, kernel, kernel, border_mode='same'))

    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(MaxPooling2D(pool_size=(pz, pz)))
    #after 4th maxpool [22.5,30]

    may_segnet.add(Convolution2D(512, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(Convolution2D(512, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(Convolution2D(512, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(MaxPooling2D(pool_size=(pz, pz)))
    #after 5th maxpool [11.25,15]
    #decoding layer
    
    may_segnet.add(UpSampling2D(size=(pz,pz)))
    may_segnet.add(Convolution2D(512, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(Convolution2D(512, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(Convolution2D(512, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    
    may_segnet.add(UpSampling2D(size=(pz,pz)))
    may_segnet.add(Convolution2D(512, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(Convolution2D(512, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(Convolution2D(256, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(UpSampling2D(size=(pz,pz)))
    may_segnet.add(Convolution2D(256, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(Convolution2D(256, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(Convolution2D(128, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(UpSampling2D(size=(pz,pz)))
    may_segnet.add(Convolution2D(128, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(Convolution2D(64, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add( Activation('relu'))
    may_segnet.add(UpSampling2D(size=(pz,pz)))
    may_segnet.add(Convolution2D(64, kernel, kernel, border_mode='same'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Activation('relu'))
    may_segnet.add(Convolution2D(classn, 1, 1, border_mode='valid'))
    may_segnet.add(BatchNormalization())
    may_segnet.add(Reshape((classn, iheight * iwidth), input_shape=(12,iheight, iwidth)))
    may_segnet.add(Permute((2, 1)))
    may_segnet.add(Activation('softmax'))
    
    return may_segnet
  
    


with open('segnetw.json', 'w') as outfile:
    modelbuild()
    outfile.write(json.dumps(json.loads(may_segnet.to_json()), indent=2))






   


##
##
##may_segnet.encoding_layers = encodnet
##for layer in encodnet:
##
##    may_segnet.add(layer)
##
##
##segnet_basic.decoding_layers = decodnet
##for layer in decodnet:
##    may_segnet.add(layer)
##


