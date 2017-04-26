import numpy as np
import cPickle

import os
import argparse
import random

"""
Creates threee text files. One for validation another one for tests. and one for all images 
Those test files contain the raw images relative path to be used for the construction of the lmdb dataset

FIRST - Builds one general images.txt with all the path to the raw images
SECOUND - Shuffle file lines
THIRD - Take the images.txt and splits into validation and testing. Default 20 percent for validation
"""
ap = argparse.ArgumentParser()
ap.add_argument("-i","--images_path",required=True,help="Raw Images Path")
args = vars(ap.parse_args())



def buildTxtFiles(imagesPath,f="images.txt"):
	with open(f,"w") as file_txt:
		for (i,image) in enumerate(os.listdir(imagesPath)):
			fileToWrite = "images.txt"			
			if image.split('_')[0] == "fire":
				label = 1
			else:
				label = 0
			file_txt.write("{} {}\n".format(args["images_path"]+image,label))			
	schuffleFileLines(f)
	splitTrainTest(f)


def splitTrainTest(f,test=0.2):	
	print("[INFO] Splitting with {}% for testing".format(test*100))
	with open(f,"r") as global_file:
		data = [line for line in global_file]
	
	test_num = int(len(data) * test)
	train_num = len(data) - test_num

	train_data = data[0:train_num]
	test_data = data[-test_num:]
	
	with open("train.txt","w") as train_file:
		for line in train_data:
			train_file.write(line)
	print("[INFO] Train.txt file with {} lines".format(len(train_data)))
	with open("val.txt","w") as test_file:
		for  line in test_data:
			test_file.write(line)
	print("[INFO] Val.txt file with {} lines".format(len(test_data)))
	
def schuffleFileLines(f):
	try:
		with open(f,"r") as file_txt:
			data = [(random.random(),line) for line in file_txt]
		data.sort()
		print("[INFO] Total of {} images".format(len(data)))	
		with open(f,"w") as file_txt:
			for _, line in data:
				file_txt.write(line)
	except Exception as e:
		print("[BUILD TXT DATASET] Error {}".format(e))
	finally:
		print("[INFO] File lines schuffled. Will split train test")			

buildTxtFiles(args["images_path"])
