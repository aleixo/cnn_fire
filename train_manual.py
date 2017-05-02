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
#from googlemanager import GoogleManager
import matplotlib
import matplotlib.pyplot as plt
from theano import function

from keras.models import Model
K.set_image_dim_ordering('th')

#ap.add_argument("-d","--imagesdir",required=True,help="Where images are")
#args = vars(ap.parse_args())

img_rows, img_cols = 64, 64
batch_size = 32
nb_classes = 2
nb_epoch = 60
nb_filters = 32
nb_pool = 2
nb_conv = 1
img_channels = 1
model = Sequential()
save_weights_filename = "weights.h5"
save_model_filename = "model.json"
#imagesPath = "Desktop/small_dataset/"
imagesPath = "Desktop/full_dataset/"
#imagesPath = args["imagesdir"]

fireLabelsNum = 0
forrestLabelsNum = 0
 
def modelArchTwo():

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
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    
def karpathyNet(numChannels, imgRows, imgCols, numClasses, **kwargs):

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
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

def leNet(numChannels, imgRows, imgCols, numClasses, activation="tanh"):

    model.add(Convolution2D(20, 5, 5, border_mode="same",input_shape=(numChannels, imgRows, imgCols)))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation(activation))
    model.add(Dense(numClasses))
    model.add(Activation("softmax"))
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

def miniVGGNet(numChannels, imgRows, imgCols, numClasses):

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
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



#Get list os all the images im array
imlist = []
for item in os.listdir(imagesPath):
    if not item.startswith("."):
        imlist.append(item)

num_samples=size(imlist)

#im1 = cv2.imread(imagesPath+imlist[1])
#m,n = im1.shape[0:2]
#imnbr = len(imlist)

#Create matrix of images. One image per line
immatrix = []
for im2 in imlist:
    if not im2.startswith("."):
        image = cv2.cvtColor(cv2.imread(imagesPath + im2), cv2.COLOR_BGR2GRAY)        
        immatrix.append(np.array(image).flatten())  

labels=np.ones((num_samples,),dtype = int)
for i,item in enumerate(imlist):
    if item.split("-")[0] == "fire":
        labels[i] = 1   
        fireLabelsNum += 1
    elif item.split("-")[0] == "forrest":
        labels[i] = 0
        forrestLabelsNum += 1
    else:        
        print("[TRAIN_MANUAL] Error building dataset. Naming convention is not followed on item {}.".format(item))
        print("[TRAIN_MANUAL] Eg of naming -> fire-12312.png")
        print("[TRAIN_MANUAL] Example only searching for fire and forrest keywords")        

print("[TRAIN_MANUAL] There are {} fire labels".format(fireLabelsNum))
print("[TRAIN_MANUAL] There are {} forrest labels".format(forrestLabelsNum))
assert len(labels) == len(immatrix)

data,Label = shuffle(immatrix,labels, random_state=2)
train_data = [data,Label]

(X, y) = (train_data[0],train_data[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

X_train = np.array(X_train)
X_test = np.array(X_test)

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print(X_train.shape)
print(X_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#cv2.imshow("{}".format(Y_train[i,0]),X_train[i, 0])
#cv2.waitKey(0)
#print("label : ", Y_train[i,:])


earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto')
miniVGGNet(1,64,64,2)

#hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoh, verbose=1, callbacks=[earlyStopping], validation_split=0.2)
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=0.2)

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)
p=model.predict_proba(X_test)

def debugConfusionMatrixAndClassificationReport():

    target_names = ['Forrest', 'Fire']
    print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))
    print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

def classifyTestImages(numImages = 20):
    if numImages < 3:
        print("[TRAIN MANUAL] Please try to classify more than three images. Easier programming :)")
        return
    
    classNames = ["Forrest","Fire"]
    
    score = model.evaluate(X_test, Y_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    preds = model.predict_classes(X_test[1:numImages])
    fig=plt.figure(figsize=(10,10))
    for i in range(numImages):

        ax = fig.add_subplot(5, 5, i+1)
        input_image=X_test[i:i+1,:,:,:]
        ax.imshow(input_image[0,0,:,:],cmap=matplotlib.cm.gray)
        ax = plt.ylabel("Previsao {}".format(classNames[preds[i - 1]]))
        ax = plt.xlabel("Atual {}".format(classNames[int(Y_test[i][1])]))
        plt.tight_layout()
    plt.show()  
    
            
debugConfusionMatrixAndClassificationReport()
classifyTestImages(25)

def debugLayer(outputLayer=1):
    
    if isinstance(model.layers[outputLayer],Flatten) or isinstance(model.layers[outputLayer],Dense):
        print("[TRAIN MANUAL] Tried to debug Flatten or Dense layer. Returning since there are no more convolutions")
        return False
    
    input_image=X_train[0:1,:,:,:]
    print("[TRAIN MANUAL] Debugging since first layer")
    print("[TRAIN MANUAL] Debugging till -> {} in position {}".format(model.layers[outputLayer],outputLayer))
    get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[outputLayer].output])
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
    
    
for i,layer in enumerate(model.layers):
    
    if not debugLayer(i):
        break

def visualizeLossAndAcc():
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(nb_epoch)

    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('Epocas')
    plt.ylabel('Perda')
    plt.title('Perda de treino vs Perda de validacao')
    plt.grid(True)
    plt.legend(['Treino','Validacao'])
    print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['seaborn-white'])
    
    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('Epocas')
    plt.ylabel('Exatidao')
    plt.title('Exatidao de treino vs Exatidao de validacao')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['seaborn-white'])
visualizeLossAndAcc()
def plotDebug():

    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


plotDebug()
def save_nn_localy():

    model_json = model.to_json()
    with open(save_model_filename, "w") as json_file:
        json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))
    model.save_weights(save_weights_filename)
    print("[TRAIN] Saved temporarily weights and model files.")
    
def upload_files():
    save_nn_localy()
    drive = GoogleManager()
    drive.init_for_upload(save_model_filename,save_model_filename)
    drive.init_for_upload(save_weights_filename,save_weights_filename)
    drive.init_for_list()
    print("[TRAIN] Removed temporarily weights and model files.")
    os.remove(save_weights_filename)
    os.remove(save_model_filename)
    