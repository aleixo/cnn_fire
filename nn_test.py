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
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import simplejson
import matplotlib
import os
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
		labels = ["Nao fogo","fogo"]
		self.downloadParams();
		return None

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
			loaded_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy', self.mean_pred])			
			loaded_model.load_weights(model_weights)
			loaded_model.summary()

		else:
			print("[NNTEST] Mandatory files architecture or weigths not present in google drive")

	def areFilesAvailable(self):
		return (os.path.exists(model_architecture) and os.path.exists(model_weights))

	def getWeights(self):
		return model_weights

	def getArch(self):
		return model_architecture

	def predictOnImage(self,image):	

		orig = image
		#image = image_utils.load_img(args["image"], target_size=(32, 32))
		image = cv2.resize(image,(64,64))
		#image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		image = image_utils.img_to_array(image)
		image = np.expand_dims(image, axis=0)
		global loaded_model		
		preds = loaded_model.predict(image)
		
		class_result=np.argmax(preds,axis=-1)
		print(int(preds[0][0]))
		global labels
		cv2.putText(orig, "Res: {}".format(labels[int(preds[0][0])]), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

		return orig,preds

	def predictOnImageDir(self,imageDir):	

		global loaded_model		
		global labels
	
		image = cv2.imread(imageDir)
		cv2.imshow("dsa",image)
		
		orig = image

		image = cv2.resize(image,(64,64))
		#image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		image = image_utils.img_to_array(image)
		image = np.expand_dims(image, axis=0)

		preds = loaded_model.predict(image)
		
		class_result=np.argmax(preds,axis=-1)

		cv2.putText(orig, "Res: {}".format(labels[int(preds[0][0])]), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
		
		cv2.imshow("Prediction",orig)
		cv2.waitKey(0)

	def processImage(self,image):

		orig = image
		image = cv2.resize(image,(64,64))
		#image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		image = image_utils.img_to_array(image)
		image = np.expand_dims(image, axis=0)
		preds = loaded_model.predict(image)

		cv2.putText(orig, "Res: {}".format(labels[int(preds[0][0])]), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
		return orig

	#Predict on one video
	def movieVal(self):

		video_output = "./videos/video_result.mp4"
		clip = VideoFileClip("./videos/video.mp4")
		clip_v = clip.fl_image(self.processImage)
		clip_v.write_videofile(video_output,audio=False)

	@staticmethod
	def debugConfusionMatrixAndClassificationReport(model,X_test,Y_test):
			
		Y_pred = model.predict(X_test)
		y_pred = np.argmax(Y_pred, axis=1)
		p=model.predict_proba(X_test)
		print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=['Forrest', 'Fire']))
		print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

	@staticmethod
	def manualConfusionMatrix(model,pathToTestImages):

		fireNeg = 0
		firePos = 0

		forrestNeg = 0
		forrestPos = 0

		labels = ["fire","forrest"]

		for img in os.listdir(pathToTestImages):
			if not img.startswith("."):				
				print("[NN TEST] Loading image {} in path {} to make predictions".format(img,pathToTestImages))
				image = cv2.imread(pathToTestImages+"/"+img)

				image = cv2.resize(image,(64,64))
				image = image_utils.img_to_array(image)
				image = np.expand_dims(image, axis=0)
				pred = model.predict(image)
				print(model.predict_classes(image))
				labelTruth = img.split("-")[0]				
				indexPredicted = int(pred[0][0])

			#Non fire
				if labelTruth == "fire":

					if labels[indexPredicted] == "fire":
						firePos += 1
					else:
						fireNeg += 1

			#fire
				elif labelTruth == "forrest":

					if labels[indexPredicted] == "forrest":
						forrestPos += 1
					else:
						forrestNeg += 1

		confusion_matrix_fire = [firePos,fireNeg]
		confusion_matrix_forrest = [forrestNeg,forrestPos]
		
		print(confusion_matrix_fire)
		print(confusion_matrix_forrest)

	@staticmethod
	def visualizeLossAndAcc(nb_epochs,hist):
		train_loss=hist.history['loss']
		val_loss=hist.history['val_loss']
		train_acc=hist.history['acc']
		val_acc=hist.history['val_acc']

		xc = range(nb_epochs)

		plt.figure(1,figsize=(7,5))
		plt.plot(xc,train_loss)
		plt.plot(xc,val_loss)
		plt.xlabel('Epocas')
		plt.ylabel('Perda')
		plt.title('Perda de treino vs Perda de validacao')
		plt.grid(True)
		plt.legend(['Treino','Validacao'])
		plt.style.use(['seaborn-white'])
		plt.show()

		plt.figure(2,figsize=(7,5))
		plt.plot(xc,train_acc)
		plt.plot(xc,val_acc)
		plt.xlabel('Epocas')
		plt.ylabel('Exatidao')
		plt.title('Exatidao de treino vs Exatidao de validacao')
		plt.grid(True)
		plt.legend(['train','val'],loc=4)
		plt.style.use(['seaborn-white'])
		plt.show()


	@staticmethod
	def classifyTestImages(model,X_test,Y_test,labels_classification,numImages = 20):
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

	@staticmethod
	def upload_files(model,save_model_filename,save_weights_filename):
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

	@staticmethod
	def debugLayer(model,X_train,outputLayer=1):
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
			ax.imshow(output_image[0,:,:,i],cmap=matplotlib.cm.gray)
        	plt.tight_layout()
    	plt.show()

    	
