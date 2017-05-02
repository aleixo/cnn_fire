from utils import Utils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dir",required=True,help="Dir where images are.")
args = vars(ap.parse_args())

utilMethods = Utils()
utilMethods.convertImagesInDirToGrayscale(args['dir'])