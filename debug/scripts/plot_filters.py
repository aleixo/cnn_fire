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
from nn_test import NnTest
classifier = NnTest()

print(classifier.get_weights())
filters = layer.W.get_value()
fig = plt.figure()
for i in range(len(filters)):
    ax = fig.add_subplot(y,x,i+1)
    ax.matshow(filters[i][0],cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
plt.tight_layout()
