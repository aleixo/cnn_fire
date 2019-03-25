import keras
import simplejson
import os
import argparse
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,ProgbarLogger,CSVLogger
from sklearn.metrics import confusion_matrix

# Own modules
from googlemanager import GoogleManager
from cnn_architectures import CnnArchitectures
from nn_test import NnTest
from raw_images_loader import RawImagesLoader

ap = argparse.ArgumentParser()
ap.add_argument("-a","--arch",required=True,help="Model arch")
ap.add_argument("-t","--testpath",required=True,help="Model arch")
args = vars(ap.parse_args())

save_weights_filename = "weights.h5"
save_model_filename = "model.json"

train_images_path = "epia_2019/images/train"
validate_images_path = "epia_2019/images/validate"
test_images_path = "epia_2019/images/test_mixed_classes"

batch_size = 16
epochs = 1
img_size = 64
forRGB = True
channels = 0
if forRGB:
    channels = 3
else:
    channels = 1
    
best_model_checkpointed="best-trainned-model.h5"
K.set_image_dim_ordering('th')

if args["arch"] == "karpathy":
        print("[TRAIN] Training with karpathy")
        model = CnnArchitectures.karpathyNet(channels,img_size,img_size,2)
elif args["arch"] == "custom":
        print("[TRAIN] Training with custom arch")
        model = CnnArchitectures.smallCustomArch(channels,img_size,img_size,2)
elif args["arch"] == "vgg":
        print("[TRAIN] Training with mini vgg")
        model = CnnArchitectures.miniVGGNet(channels,img_size,img_size,2)
elif args["arch"] == "lenet":
        print("[TRAIN] Training with mini leNet")
        model = CnnArchitectures.leNet(channels,img_size,img_size,2)
elif args["arch"] == "mlp":
    print("[TRAIN] Training with mlp")
    model = CnnArchitectures.mlp(2,img_size)

images_loader = RawImagesLoader()

X_test,Y_test = images_loader.getImageRepresentationManualSplit(
    args["testpath"],
    img_size,
    2,
    forRGB=forRGB
    )

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_images_path,
    target_size=(img_size, img_size),
    batch_size=batch_size
    )  

validation_generator = test_datagen.flow_from_directory(
        validate_images_path,
        target_size=(img_size, img_size),
        batch_size=batch_size      
        )

checkpoint = ModelCheckpoint(
    best_model_checkpointed, 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    mode='min'
    )
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=2, 
    cooldown=1,
    min_lr=0.0001,
    factor=0.5,
    verbose=1,
    mode='auto')
train_callbacks = [checkpoint]

print(train_generator.class_indices)
#print(validation_generator.class_indices)

history = model.fit_generator(
        train_generator,
        steps_per_epoch=1,
        epochs=epochs,
        verbose=1,
        callbacks=train_callbacks,
        validation_data=validation_generator,
        validation_steps=1
)

model.load_weights(best_model_checkpointed)
os.remove(best_model_checkpointed)
NnTest.debugConfusionMatrixAndClassificationReport(model,X_test,Y_test)
NnTest.visualizeLossAndAcc(epochs,history)
NnTest.upload_files(model,save_model_filename,save_weights_filename)
nnTest = NnTest() 
nnTest.predictOnImageDir(args["testpath"])	



