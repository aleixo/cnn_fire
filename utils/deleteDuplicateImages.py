#coding:utf-8
from imutils import paths
import cv2
import numpy as np
import os
import subprocess
import glob
import  matplotlib.pyplot as plt
import matplotlib.pylab as pltlab
import random

"""
DESCRIPTION
Utils used by other scripts
-resizeImages - Resizes images 32,32
-changeImgNames - Changes name of images from directory
-shuffle_in_unisson - Shuffles two arrays using the same randomizer
-rawToNumpy - Converts raw image format into numpy array
-deleteDuplicated - Deletes duplicated images from directory
-getTrainAndTest - Splits train and test data


Call to delete duplicated on di
from utils import Utils
utilMethods = Utils()
utilMethods.deleteDuplicated(destination_dir)
utilMethods.renameFilesInFolder("t","forrest")

"""

class Utils(object):

	colors = ('b', 'g', 'r')

	def resizeImages(path):
		print(path)
		directory = path
		try:
			for imagef in os.listdir(directory):
				if imagef == ".DS_Store":
					pass
				else:
					img = os.path.join(directory,imagef)
					image = cv2.resize(cv2.imread(img),(32,32),interpolation = cv2.INTER_AREA)
					cv2.imwrite(img,image)
		except Exception as e:
			print("ERROR {}".format(e))
		finally:t
		print("[INFO] Images with 32*32")
	def changeImgNames():
		directory = "/Users/diogoaleixo/Desktop/Mestrado/images_raw/images/"
		try:
			for i,imagef in enumerate(os.listdir("/Users/diogoaleixo/Desktop/Mestrado/images_raw/images/")):
				if imagef == ".DS_Store":
					pass
				else:
					img = os.path.join(directory,imagef)
					if imagef.split('_')[0] == "fire":
						os.rename(img,"fire_"+str(i)+".jpg")
					else:
						os.rename(img,"forrest_"+str(i)+".jpg")
		except Exception as e:
			print("ERROR {}".format(e))
		finally:
			print("[INFO] Image names changed")


	def shuffle_in_unison(a, b):

		assert len(a) == len(b)
		shuffled_a = np.empty(a.shape, dtype=a.dtype)
		shuffled_b = np.empty(b.shape, dtype=b.dtype)
		permutation = np.random.permutation(len(a))
		for old_index, new_index in enumerate(permutation):
			shuffled_a[new_index] = a[old_index]
			shuffled_b[new_index] = b[old_index]
		return shuffled_a, shuffled_b

	def deleteDuplicated(self,destination_dir):	

		colors = ('b','g','r')
		hists = []
		hists2 = []
		indexInComparison = 0
		duplicate = 0	
		for i,imagePath in enumerate(paths.list_images(destination_dir)):		
			image = cv2.imread(imagePath)
			channels = cv2.split(image)
			indexInComparison = i
		
			for (channel,color) in zip(channels,colors):

				hist = cv2.calcHist([channel],[0],None,[256],[0,256])	
				hists.append(hist)
		
			for i,imagePath2 in enumerate(paths.list_images(destination_dir)):
				if i > indexInComparison:				
					image2 = cv2.imread(imagePath2)
					channels2 = cv2.split(image2)
					for (channel2,color2) in zip(channels2,colors):
						hist2 = cv2.calcHist([channel2],[0],None,[256],[0,256])	
						hists2.append(hist2)					
					for (h1,h2) in zip(hists,hists2):					
						res = cv2.compareHist(h1,h2,cv2.cv.CV_COMP_BHATTACHARYYA)	
						if res == 0.0:
							duplicate += 1									
					if duplicate == 3:
						print("[INFO] IGUAIS {} {}".format(imagePath,imagePath2))
						print("[INFO] Eliminar {}".format(imagePath2))
						os.remove(imagePath2)										
					duplicate = 0
					hists2 = []
			hists = []

	def plot3DHistogram(self,image1,image2,saveDIr):

		for i,color in enumerate(self.colors):

			hist = cv2.calcHist([image1], [i], None, [256], [0, 256])
			ax1 = plt.subplot(211)
			ax1 = plt.plot(hist, color=color)
			ax1 = plt.xlim([0, 256])


		for i,color in enumerate (self.colors):

			hist = cv2.calcHist([image2], [i], None, [256], [0, 256])
			ax2 = plt.subplot(212)
			ax2 = plt.plot(hist, color=color)
			ax2 = plt.xlim([0, 256])

		#plt.show()
		pltlab.savefig("Histograms/hist_"+str(random.uniform(0.0,10000.0))+".png")

	def deleteDebugImages(self):
		for filename in glob.glob("Histograms/hist_*"):
			os.remove(filename)

	def renameFilesInFolder(self,destination_dir,prefix):
  		for filename in os.listdir(destination_dir):
  			print(filename)
  			print(destination_dir + "/" + prefix+"-"+str(random.uniform(0.0,10000.0))+".png")
  			os.rename(destination_dir + "/" + filename,destination_dir + "/" + prefix+"-"+str(random.uniform(0.0,10000.0))+".png")