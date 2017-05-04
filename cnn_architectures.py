from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import keras
import argparse
import numpy as np
import os
import theano
from numpy import *
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
ap = argparse.ArgumentParser()
import cv2
import simplejson
from keras import backend as K
from sklearn.metrics import classification_report,confusion_matrix
from googlemanager import GoogleManager
import matplotlib
import matplotlib.pyplot as plt
from theano import function
from keras.models import Model


class CnnArchitectures:            

    def modelArchTwo(self):

        model = Sequential()
        model.add(Convolution2D(16, (3, 3),input_shape=(1, 64, 64)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    
        model.add(Convolution2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    
        model.add(Convolution2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
        model.add(Flatten())  
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
        return model
    
    def karpathyNet(self,numChannels, imgRows, imgCols, numClasses, **kwargs):

        model = Sequential()
        model.add(Convolution2D(16, 5, 5, border_mode="same",input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Convolution2D(20, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))
    
        model.add(Convolution2D(20, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))
    
        model.add(Flatten())
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))
        model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
        return model

    def leNet(self,numChannels, imgRows, imgCols, numClasses, activation="tanh"):

        model = Sequential()
        model.add(Convolution2D(20, 5, 5, border_mode="same",input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))
        model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
        return model

    def miniVGGNet(self,numChannels, imgRows, imgCols, numClasses):

        model = Sequential()
        model.add(Convolution2D(32, 3, 3, border_mode="same",input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation("relu"))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))  
    
        model.add(Convolution2D(64, 3, 3, border_mode="same"))
        model.add(Activation("relu"))
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
    
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))
        model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
        return model
