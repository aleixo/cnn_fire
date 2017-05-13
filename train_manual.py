from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,ProgbarLogger,CSVLogger
import keras

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import StratifiedKFold

import matplotlib
import matplotlib.pyplot as plt

from numpy import size
import numpy as np

import theano
import cv2
import simplejson
import argparse
import os

from raw_images_loader import RawImagesLoader
from cnn_architectures import CnnArchitectures
from googlemanager import GoogleManager
from nn_test import NnTest

K.set_image_dim_ordering('th')

ap = argparse.ArgumentParser()
ap.add_argument("-d","--imagesdir",required=True,help="Where images are")
args = vars(ap.parse_args())

labels_classification = ['Forrest', 'Fire']
save_weights_filename = "weights.h5"
save_model_filename = "model.json"
images_path = args["imagesdir"]
best_model_checkpointed="best-trainned-model.h5"

img_size = 128
is_rgb = True
img_channels = 3
nb_epochs = 60

architectures = CnnArchitectures()
images_loader = RawImagesLoader()

X_train,Y_train,X_test,Y_test,X_val,Y_val = images_loader.getImagesRepresentation(
    images_path,
    img_size,
    size(labels_classification),
    forRGB=is_rgb,
    test_size=0.15,
    val_size=0.35
    )

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    min_delta=0, 
    patience=10, 
    verbose=0, 
    mode='auto'
    )

checkpoint = ModelCheckpoint(
    best_model_checkpointed, 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    mode='min'
    )

tensorboard = keras.callbacks.TensorBoard(
    log_dir='./Graph', 
    histogram_freq=1,  
    write_graph=True, 
    write_images=True,

    )

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=4, 
    cooldown=1,
    min_lr=0.0001,
    factor=0.9,
    verbose=1,
    mode='auto')

csv_logger = CSVLogger('training.log')
model = architectures.miniVGGNet(img_channels,img_size,img_size,size(labels_classification))
train_callbacks = [checkpoint,reduce_lr,csv_logger]

hist = model.fit(
    X_train, 
    Y_train, 
    batch_size=64, 
    nb_epoch=nb_epochs, 
    verbose=1, 
    callbacks=train_callbacks, 
    validation_data=(X_val,Y_val)
    )

def debugAllLayers():
    for i,layer in enumerate(model.layers):    
        if not NnTest.debugLayer(model,X_train,i):
            break   

def evaluateAndDebug(model):
    print("[TRAIN MANUAL] Will valuate model")
    scoreTest = model.evaluate(X_train,Y_train)
    scoreVal = model.evaluate(X_val,Y_val)
    print("[TRAIN MANUAL] Score meaning --> {}".format(model.metrics_names))
    print("[TRAIN MANUAL] Score on test set {} and {} on val set".format(scoreTest,scoreVal))

print("[TRAIN MANUAL] Last epoch confusion matrix and score")
NnTest.debugConfusionMatrixAndClassificationReport(model,X_test,Y_test)
evaluateAndDebug(model)

print("[TRAIN MANUAL] Best epoch confusion matrix and score")
model.load_weights(best_model_checkpointed)
os.remove(best_model_checkpointed)
NnTest.debugConfusionMatrixAndClassificationReport(model,X_test,Y_test)
evaluateAndDebug(model)

NnTest.classifyTestImages(model,X_test,Y_test,labels_classification, 25)
NnTest.visualizeLossAndAcc(nb_epochs,hist)
NnTest.upload_files(model,save_model_filename,save_weights_filename)





    