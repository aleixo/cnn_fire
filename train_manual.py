from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,ProgbarLogger,CSVLogger
import keras
from keras.preprocessing.image import ImageDataGenerator
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


#python train_manual.py -d ~/Desktop/datasets/rgb_full_dataset_separated/train/ -t ~/Desktop/datasets/rgb_full_dataset_separated/test/ -v ~/Desktop/datasets/rgb_full_dataset_separated/val/

K.set_image_dim_ordering('th')

ap = argparse.ArgumentParser()
ap.add_argument("-d","--imagesdirtrain",required=True,help="Where images are")
ap.add_argument("-t","--imagesdirtest",required=False,help="Where test images are")
ap.add_argument("-v","--imagesdirval",required=False,help="Where test images are")

args = vars(ap.parse_args())

labels_classification = ['Forrest', 'Fire']
save_weights_filename = "weights.h5"
save_model_filename = "model.json"
images_path_train = args["imagesdirtrain"]
images_path_test = args["imagesdirtest"]
images_path_val = args["imagesdirval"]
best_model_checkpointed="best-trainned-model.h5"

img_size = 128
is_rgb = True
img_channels = 3
nb_epochs = 60

if is_rgb:
    img_channels = 3
else:
    img_channels = 1

test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
        args["imagesdirval"],
        target_size=(img_size, img_size),
        batch_size=8,
        save_to_dir="/Users/diogoaleixo/Desktop/datasets/augmented/",
        save_prefix="png"
        )

architectures = CnnArchitectures()
images_loader = RawImagesLoader()

print("[TRAIN MANUAL] Loading train dataset")

X_train,Y_train = images_loader.getImageRepresentationManualSplit(
    images_path_train,
    img_size,
    size(labels_classification),
    forRGB=is_rgb
    )

print("[TRAIN MANUAL] Loading validation dataset")

X_val,Y_val = images_loader.getImageRepresentationManualSplit(
    images_path_val,
    img_size,
    size(labels_classification),
    forRGB=is_rgb
    )

print("[TRAIN MANUAL] Loading test dataset")

X_test,Y_test = images_loader.getImageRepresentationManualSplit(
    images_path_test,
    img_size,
    size(labels_classification),
    forRGB=is_rgb
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
    patience=2, 
    cooldown=2,
    min_lr=0.0001,
    factor=0.5,
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





    