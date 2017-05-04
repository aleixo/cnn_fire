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
from raw_images_loader import RawImagesLoader
from cnn_architectures import CnnArchitectures
from keras.models import Model
K.set_image_dim_ordering('th')

ap.add_argument("-d","--imagesdir",required=True,help="Where images are")
args = vars(ap.parse_args())

labels_classification = ['Forrest', 'Fire']
save_weights_filename = "weights.h5"
save_model_filename = "model.json"
images_path = args["imagesdir"]
img_size = 64
is_rgb = True
img_channels = 3
nb_epochs = 1
#images_path = "Desktop/small_dataset/"
#images_path = "Desktop/datasets/full_dataset/"


images_loader = RawImagesLoader()
X_train,Y_train,X_test,Y_test = images_loader.getImagesRepresentation(images_path,img_size,size(labels_classification),forRGB=is_rgb)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto')
architectures = CnnArchitectures()
model = architectures.miniVGGNet(img_channels,img_size,img_size,size(labels_classification))
    #hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoh, verbose=1, callbacks=[earlyStopping], validation_split=0.2)
hist = model.fit(X_train, Y_train, batch_size=32, nb_epoch=nb_epochs, verbose=1, validation_split=0.2)

def debugConfusionMatrixAndClassificationReport():
    
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    p=model.predict_proba(X_test)

    print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=labels_classification))
    print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

def classifyTestImages(numImages = 20):
    if numImages < 3:
        print("[TRAIN MANUAL] Please try to classify more than three images. Easier programming :)")
        return

    score = model.evaluate(X_test, Y_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    preds = model.predict_classes(X_test[1:numImages])
    fig=plt.figure(figsize=(10,10))
    for i in range(numImages):

        ax = fig.add_subplot(5, 5, i+1)
        input_image=X_test[i:i+1,:,:,:]
        ax.imshow(input_image[0,0,:,:],cmap=matplotlib.cm.gray)
        ax = plt.ylabel("Previsao {}".format(labels_classification[preds[i - 1]]))
        ax = plt.xlabel("Atual {}".format(labels_classification[int(Y_test[i][1])]))
        plt.tight_layout()
    plt.show()  
    
            
def debugLayer(outputLayer=1):
    
    if isinstance(model.layers[outputLayer],Flatten) or isinstance(model.layers[outputLayer],Dense):
        print("[TRAIN MANUAL] Tried to debug Flatten or Dense layer. Returning since there are no more convolutions")
        return False
    
    input_image=X_train[0:1,:,:,:]
    print("[TRAIN MANUAL] Debugging since first layer")
    print("[TRAIN MANUAL] Debugging till -> {} in position {}".format(model.layers[outputLayer],outputLayer))
    get_3rd_layer_output = K.function([model.layers[0].input],[model.layers[outputLayer].output])
    output_image = get_3rd_layer_output([X_train])[0]
    output_image = np.rollaxis(np.rollaxis(output_image, 3, 1), 3, 1)
    fig=plt.figure(figsize=(8,8))

    for i in range(16):
        if i == 0:
            ax = fig.add_subplot(6, 6, 1)
            
            ax.imshow(input_image[0,0,:,:],cmap=matplotlib.cm.gray)
        ax = fig.add_subplot(6, 6, i+7)
        #ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
        ax.imshow(output_image[0,:,:,i],cmap=matplotlib.cm.gray)
        plt.tight_layout()
    plt.show()
    return True

def debugAllLayers():
    for i,layer in enumerate(model.layers):    
        if not debugLayer(i):
            break   

def visualizeLossAndAcc():
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(nb_epochs)

    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('Epocas')
    plt.ylabel('Perda')
    plt.title('Perda de treino vs Perda de validacao')
    plt.grid(True)
    plt.legend(['Treino','Validacao'])
    plt.style.use(['seaborn-white'])
    
    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('Epocas')
    plt.ylabel('Exatidao')
    plt.title('Exatidao de treino vs Exatidao de validacao')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    plt.style.use(['seaborn-white'])
    
def uploadFiles():

    model_json = model.to_json()
    with open(save_model_filename, "w") as json_file:
        json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))
    model.save_weights(save_weights_filename)
    print("[TRAIN] Saved temporarily weights and model files.")

    drive = GoogleManager()
    drive.init_for_upload(save_model_filename,save_model_filename)
    drive.init_for_upload(save_weights_filename,save_weights_filename)
    drive.init_for_list()
    print("[TRAIN] Removed temporarily weights and model files.")
    os.remove(save_weights_filename)
    os.remove(save_model_filename)


debugConfusionMatrixAndClassificationReport()
#classifyTestImages(25)
#visualizeLossAndAcc()
#uploadFiles()
#debugAllLayers()

    