import keras
import argparse
import numpy as np
import os
import theano
import cv2
import simplejson
import matplotlib
import matplotlib.pyplot as plt

from theano import function
from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from numpy import *
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

from googlemanager import GoogleManager

class CnnArchitectures:     

    @staticmethod
    def smallCustomArch(numChannels, imgRows, imgCols, numClasses):

        model = Sequential()
        model.add(Convolution2D(32, (3, 3), input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())  
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(numClasses))
        model.add(Activation('softmax'))

        adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

        model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

        return model

    @staticmethod
    def karpathyNet(numChannels, imgRows, imgCols, numClasses):

        model = Sequential()
        model.add(Convolution2D(16, 5, 5, border_mode="same",input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Convolution2D(20, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))
    
        model.add(Convolution2D(20, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))
    
        model.add(Flatten())
        model.add(Dense(numClasses))
        model.add(Activation("sigmoid"))
        model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

        return model

    @staticmethod
    def leNet(numChannels, imgRows, imgCols, numClasses, activation="tanh"):

        model = Sequential()
        model.add(Convolution2D(20, 5, 5, border_mode="same",input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))
        model.add(Dense(numClasses))
        model.add(Activation("sigmoid"))
        model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

        return model

    @staticmethod
    def miniVGGNet(numChannels, imgRows, imgCols, numClasses):

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
        
        adadelta = keras.optimizers.Adadelta(lr=0.1, rho=0.95, epsilon=1e-08, decay=0.0)

        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
        
        return model
    @staticmethod
    def mlp(numClasses,imgRows):

        shape = imgRows*imgRows

        model = Sequential()
        model.add(Dense(128,activation='relu', input_shape=(128,2)))
        model.add(Dropout(0.5))
        model.add(Dense(128,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(numClasses,activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='Adam', matrics = ['accuracy'])
        return model