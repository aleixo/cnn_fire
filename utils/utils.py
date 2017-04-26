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
    def shuffle_in_unison(a, b):
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        
        for old_index, new_index in enumerate(permutation):
            
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
            
        return shuffled_a, shuffled_b
    
    def deleteDuplicatedImagesInDir(self,dir):
        
        colors = ('b', 'g', 'r')
        hists = []
        hists2 = []
        indexInComparison = 0
        duplicate = 0
        initialFilesLen = len(os.listdir(dir))
        print("[UTILS] Directory has {} ".format(initialFilesLen))
        
        for i,imagePath in enumerate(paths.list_images(dir)):
            
            hists = []  
            image = cv2.imread(imagePath)
            channels = cv2.split(image)
            indexInComparison = i
            
            for (channel,color) in zip(channels,colors):
            
                hists.append(cv2.calcHist([channel],[0],None,[256],[0,256]))
                
               
            for i2,imagePath2 in enumerate(paths.list_images(dir)):
                hists2 = []
                if i2 > indexInComparison:
                
                    print("[UTILS] Checking for duplicate images in path {} and {}".format(imagePath,imagePath2))
                    
                    image2 = cv2.imread(imagePath2)
                    channels2 = cv2.split(image2)
                    
                    self.plot3DHistogram(image,image2,dir)
     
                    for (channel2,color) in zip(channels2,colors):
                        
                        hists2.append(cv2.calcHist([channel2],[0],None,[256],[0,256]))

                    duplicate = 0
                    for (h1,h2) in zip(hists,hists2):
                        
                        if len(h1) is 256 and len(h2) is 256:
                            
                            if cv2.compareHist(h1,h2,cv2.cv.CV_COMP_BHATTACHARYYA) == 0.0:
                                
                                duplicate +=1
                                
                                if duplicate == 3:
                                    print("[DELETE DUPLICATE IMAGES] Equal images {} {}".format(imagePath,imagePath2))
                                    os.remove(imagePath2)
                                    
            
            print("[UTILS] Folder had {} images and now has {}. {} were deleted".
                  format(initialFilesLen,
                         len(os.listdir(dir)),
                         initialFilesLen - len(os.listdir(dir))))
                    
                                    
                                    
    def plotImage3DHistogram(self,image,dir):
        
        print("[UTILS] Plot 3D histogram")
        for i,color in enumerate(self.colors):

            hist = cv2.calcHist([image], [1], None, [256], [0, 256])
            ax1 = plt.subplot(211)
            ax1 = plt.plot(hist, color=color)
            ax1 = plt.xlim([0, 256])

        #plt.show()
        #pltlab.savefig(dir+str(random.uniform(0.0,10000.0))+".png")

    def plot3DHistogram(self,image1,image2,saveDir):

        for i,color in enumerate(self.colors):

            hist = cv2.calcHist([image1], [i], None, [256], [0, 256])
            ax1 = plt.subplot(221)
            ax1 = plt.plot(hist, color=color)
            ax1 = plt.xlim([0, 256])
            ax1 = plt.ylabel("Histogram in comparision")
            
            
        for i,color in enumerate(self.colors):
            
            ax1 = plt.subplot(222)
            ax1 = plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
            ax1 = plt.ylabel("Image in comparision")
        
        for i,color in enumerate(self.colors):

            hist = cv2.calcHist([image2], [i], None, [256], [0, 256])
            ax1 = plt.subplot(223)
            ax1 = plt.plot(hist, color=color)
            ax1 = plt.xlim([0, 256])
            ax1 = plt.ylabel("Histogram to compara")
            
        for i,color in enumerate(self.colors):

            ax1 = plt.subplot(224)
            ax1 = plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
            ax1 = plt.ylabel("Image to compare")
            
        plt.show()
        #pltlab.savefig(saveDir+str(random.uniform(0.0,10000.0))+".png")

    def renameFilesInFolder(self,destination_dir,prefix):
        for filename in os.listdir(destination_dir):
            if not filename.startswith("."):
                os.rename(destination_dir + filename,destination_dir + prefix+"-"+str(random.uniform(0.0,10000.0))+".png")
