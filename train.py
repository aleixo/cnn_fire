from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.utils import plot_model
import keras
import simplejson
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,ProgbarLogger,CSVLogger
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import argparse
from googlemanager import GoogleManager
from cnn_architectures import CnnArchitectures
from nn_test import NnTest
from raw_images_loader import RawImagesLoader

ap = argparse.ArgumentParser()
ap.add_argument("-a","--arch",required=True,help="Model arch")
args = vars(ap.parse_args())

save_weights_filename = "weights.h5"
save_model_filename = "model.json"

train_images_path = "images/train"
validate_images_path = "images/validate"
test_images_path = "images/test_mixed_classes"

batch_size = 16
epochs = 60




best_model_checkpointed="best-trainned-model.h5"
K.set_image_dim_ordering('th')

if args["arch"] == "karpathy":
        print("[TRAIN] Training with karpathy")
        model = CnnArchitectures.karpathyNet(3,64,64,1)
elif args["arch"] == "custom":
        print("[TRAIN] Training with custom arch")
        model = CnnArchitectures.smallCustomArch(3,64,64,1)
elif args["arch"] == "vgg":
        print("[TRAIN] Training with mini vgg")
        model = CnnArchitectures.miniVGGNet(3,64,64,1)
elif args["arch"] == "lenet":
        print("[TRAIN] Training with mini leNet")
        model = CnnArchitectures.leNet(3,64,64,1)
        
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_images_path,  
        target_size=(64, 64),  
        batch_size=batch_size,
        class_mode='binary'
        )  

validation_generator = test_datagen.flow_from_directory(
        validate_images_path,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='binary'
        )

checkpoint = ModelCheckpoint(
    best_model_checkpointed, 
    monitor='val_acc', 
    verbose=1, 
    save_best_only=True, 
    mode='max'
    )

train_callbacks = [checkpoint]

#print(train_generator.class_indices)
#print(validation_generator.class_indices)


history = model.fit_generator(
        train_generator,
        steps_per_epoch=1700,
        epochs=epochs,
        verbose=1,
        callbacks=train_callbacks,
        validation_data=validation_generator,
        validation_steps=196
)

#NnTest.visualizeLossAndAcc(epochs,history)
NnTest.manualConfusionMatrix(model,test_images_path)
NnTest.upload_files(model,save_model_filename,save_weights_filename)



