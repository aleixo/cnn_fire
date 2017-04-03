from __future__ import print_function
from __future__ import absolute_import
import warnings
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import model_from_json
from keras import backend as K
from keras.preprocessing import image as image_utils
from moviepy.editor import VideoFileClip
import numpy as np
import imutils
import cv2
import argparse
from keras.utils import plot_model

"""

Test the network

Must have weights and network architecture on the same folder

"""

model_architecture = "model_more_images.json"
model_weights = "more_images2.h5"

K.set_image_dim_ordering('th')

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	help="path to the input image")
args = vars(ap.parse_args())

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


loaded_model = Sequential()
json_file = open(model_architecture, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', mean_pred])
loaded_model.load_weights(model_weights)
loaded_model.summary()

labels = ["fire","not fire"]

def predictOnImage():	

	orig = cv2.imread(args["image"])
	image = image_utils.load_img(args["image"], target_size=(32, 32))
	image = image_utils.img_to_array(image)
	image = np.expand_dims(image, axis=0)
	preds = loaded_model.predict(image)
	print(preds)
	class_result=np.argmax(preds,axis=-1)
	print(preds.astype('int'))
	cv2.putText(orig, "Res: {}".format(labels[int(preds[0][0])]), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
	cv2.imshow("Classification", orig)
	cv2.waitKey(0)

def processImage(image):

	orig = image
	image = cv2.resize(image,(32,32))
	image = image_utils.img_to_array(image)
	image = np.expand_dims(image, axis=0)
	preds = loaded_model.predict(image)

	cv2.putText(orig, "Res: {}".format(labels[int(preds[0][0])]), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
	return orig

#Predict on one video
def moviewVal():

	video_output = "./videos/video_result.mp4"
	clip = VideoFileClip("./videos/video.mp4")
	clip_v = clip.fl_image(processImage)
	clip_v.write_videofile(video_output,audio=False)


#If image arg was given, predict on it. Otherwise, try to find video to predict
if args["image"]:
	predictOnImage()
else:
	moviewVal()

