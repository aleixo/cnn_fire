from nn_test import NnTest
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=False,help="Image to predict on.")
args = vars(ap.parse_args())

classifier = NnTest()


if(args["image"]):	
	classifier.predictOnImageDir(args["image"])
else:
	classifier.movieVal()
