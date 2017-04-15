from __future__ import print_function
from __future__ import absolute_import
import warnings
import os.path
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
from google_manager import GoogleManager

"""
Test the network
Must have weights and network architecture on the same folder
"""

class NnTest:

	global model_architecture
	global model_weights
	global loaded_model
	
	model_architecture = "model.json"
	model_weights = "weights.h5"
	loaded_model = Sequential()	

	def __init__(self):			

		K.set_image_dim_ordering('th')
		global labels
		labels = ["fire","not fire"]
		self.downloadParams();	

	def mean_pred(self,y_true, y_pred):
		return K.mean(y_pred)
	
	def downloadParams(self):
		gotFiles = False
		drive = GoogleManager()
		
		print("Will download")
			

		drive.init_for_download(model_weights)	
		drive.init_for_download(model_architecture)		

		if (os.path.exists(model_architecture) and os.path.exists(model_weights)):

			global loaded_model
			json_file = open(model_architecture, 'r')
			loaded_model_json = json_file.read()
			json_file.close()
			loaded_model = model_from_json(loaded_model_json)
			loaded_model.compile(loss='binary_crossentropy',
				optimizer='rmsprop',
				metrics=['accuracy', self.mean_pred])			
			loaded_model.load_weights(model_weights)
			loaded_model.summary()

		else:
			print("[NNTEST] Mandatory files architecture or weigths not present in google drive")

	def areFilesAvailable(self):
		return (os.path.exists(model_architecture) and os.path.exists(model_weights))			

	def predictOnImage(self,image):	

		orig = image
		#image = image_utils.load_img(args["image"], target_size=(32, 32))
		image = cv2.resize(image,(32,32))
		image = image_utils.img_to_array(image)
		image = np.expand_dims(image, axis=0)
		global loaded_model		
		preds = loaded_model.predict(image)
		
		class_result=np.argmax(preds,axis=-1)
	
		global labels
		cv2.putText(orig, "Res: {}".format(labels[int(preds[0][0])]), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

		return orig,preds
		#cv2.imshow("Classification", orig)
		#cv2.waitKey(0)
