from time import time
import os,random
import cv2
import sys

sys.path.insert(0, '//Users/diogoaleixo/Desktop/cnn_fire/')
from nn_test import NnTest

class Camera(object):
    """An emulated camera implementation that streams a repeated sequence of
    files 1.jpg, 2.jpg and 3.jpg at a rate of one frame per second."""

    numImages = 0

    def __init__(self):
        self.classifier = NnTest()
        datasetPath = "/Users/diogoaleixo/Desktop/datasets/rgb_full_dataset_separated/test/"
        files = os.listdir(datasetPath)
        random.shuffle(files)
        self.frames = [open(datasetPath + file).read( ) for file in files ]
        self.imgPaths = [datasetPath + file for file in files]
        self.numImages = len(os.listdir(datasetPath))

    def get_frame(self):

        newIndex = int(time()) % self.numImages
        newFrame = self.frames[newIndex]
        im = cv2.imread(self.imgPaths[newIndex])

        temporaryImageFile = "temp.png"
        imageClassified, pred = self.classifier.predictOnImage(cv2.imread(self.imgPaths[newIndex]))
        cv2.imwrite("temp.png",imageClassified)
        self.frame = open(temporaryImageFile).read()
        return self.frame
