from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import keras
import numpy as np
import os
import theano
from numpy import *
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import cv2
import simplejson
from keras import backend as K
from sklearn.metrics import classification_report,confusion_matrix
from googlemanager import GoogleManager
import matplotlib
import matplotlib.pyplot as plt
from theano import function
from keras.models import Model


class RawImagesLoader:            

    def createImageList(self,imagesPath):
        imlist = []
        for item in os.listdir(imagesPath):
            if not item.startswith("."):
                imlist.append(item)

        num_samples=size(imlist)    
        print("[RAW IMAGES LOADER] There are {} images in total".format(num_samples))
        return imlist    

    def createImagePixelMatrix(self,imlist,imagesPath,forRGB):

        print("[RAW IMAGES LOADER] Images are RGB --> {}".format(forRGB))

        immatrix = []
        for im2 in imlist:
            if not im2.startswith("."):
                if not forRGB:
                    image = cv2.cvtColor(cv2.imread(imagesPath + im2), cv2.COLOR_BGR2GRAY)        
                else:
                    image = cv2.imread(imagesPath + im2)   
                immatrix.append(np.array(image).flatten())  
    
        return immatrix
    def createLabelsVector(self,num_samples,imlist):

        fireLabelsNum = 0;
        forrestLabelsNum = 0;

        labels=np.ones((num_samples,),dtype = int)
        for i,item in enumerate(imlist):
            if item.split("-")[0] == "fire":
                labels[i] = 0   
                fireLabelsNum += 1
            elif item.split("-")[0] == "forrest":
                labels[i] = 1
                forrestLabelsNum += 1
            else:        
                print("[TRAIN_MANUAL] Error building dataset. Naming convention is not followed on item {}.".format(item))
                print("[TRAIN_MANUAL] Eg of naming -> fire-12312.png")
                print("[TRAIN_MANUAL] Example only searching for fire and forrest keywords")        

        print("[TRAIN_MANUAL] There are {} fire labels".format(fireLabelsNum))
        print("[TRAIN_MANUAL] There are {} forrest labels".format(forrestLabelsNum))
    
        return labels

    def getImageRepresentationManualSplit(self,imagesPath,imageSize,numClasses,forRGB = False):

        print("[RAW IMAGES LOADER] Will load images from {}".format(imagesPath))

        imlist = self.createImageList(imagesPath)

        print("[RAW IMAGES LOADER] Got {} images on list".format(size(imlist)))

        immatrix = self.createImagePixelMatrix(imlist,imagesPath,forRGB)        
        labels = self.createLabelsVector(size(imlist),imlist)

        data,Label = shuffle(immatrix,labels, random_state=2)
        train_data = [data,Label]

        channels = 1
        if forRGB:
            channels = 3

        (X_test, y_test) = (train_data[0],train_data[1])

        
        X_test = np.array(X_test)            
        X_test = X_test.reshape(X_test.shape[0], channels, imageSize, imageSize)  
        X_test = X_test.astype('float32')           
        X_test /= 255  
        print("[RAW IMAGES LOADER] Loaded images with shape {} for test dataset".format(X_test.shape))
        Y_test = np_utils.to_categorical(y_test, numClasses)
        return X_test,Y_test
    def getImagesRepresentation(self,imagesPath,imageSize,numClasses,forRGB = False,test_size=0,val_size=0.2):

        print("[RAW IMAGES LOADER] Will load images from {}".format(imagesPath))

        imlist = self.createImageList(imagesPath)

        print("[RAW IMAGES LOADER] Got {} images on list".format(size(imlist)))

        immatrix = self.createImagePixelMatrix(imlist,imagesPath,forRGB)        
        labels = self.createLabelsVector(size(imlist),imlist)


        data,Label = shuffle(immatrix,labels, random_state=2)
        train_data = [data,Label]

        channels = 1
        if forRGB:
            channels = 3

        (X, y) = (train_data[0],train_data[1])

        if test_size is 0:

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=4)
            X_train = np.array(X_train)            
            X_val = np.array(X_val)

            X_train = X_train.reshape(X_train.shape[0], channels, imageSize, imageSize)            
            X_val = X_val.reshape(X_val.shape[0], channels, imageSize, imageSize)

            X_train = X_train.astype('float32')            
            X_val = X_val.astype('float32')

            X_train /= 255            
            X_val /= 255

            print("[RAW IMAGES LOADER] Loaded images with shape {} for train dataset".format(X_train.shape))
            print("[RAW IMAGES LOADER] Loaded images with shape {} for validation dataset".format(X_val.shape))                    

            Y_train = np_utils.to_categorical(y_train, numClasses)
            Y_val = np_utils.to_categorical(y_val, numClasses)

            return X_train,Y_train,X_val,Y_val

        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=4)

            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=4)

            X_train = np.array(X_train)
            X_test = np.array(X_test)
            X_val = np.array(X_val)

        

            X_train = X_train.reshape(X_train.shape[0], channels, imageSize, imageSize)
            X_test = X_test.reshape(X_test.shape[0], channels, imageSize, imageSize)
            X_val = X_val.reshape(X_val.shape[0], channels, imageSize, imageSize)

            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')
            X_val = X_val.astype('float32')

            X_train /= 255
            X_test /= 255
            X_val /= 255

            print("[RAW IMAGES LOADER] Loaded images with shape {} for train dataset".format(X_train.shape))
            print("[RAW IMAGES LOADER] Loaded images with shape {} for validation dataset".format(X_val.shape))        
            print("[RAW IMAGES LOADER] Loaded images with shape {} for test dataset".format(X_test.shape))        

            Y_train = np_utils.to_categorical(y_train, numClasses)
            Y_test = np_utils.to_categorical(y_test, numClasses)
            Y_val = np_utils.to_categorical(y_val, numClasses)

            return X_train,Y_train,X_test,Y_test,X_val,Y_val
