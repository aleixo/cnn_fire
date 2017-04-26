from utils import Utils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dir",required=True,help="Dir where images are.")
ap.add_argument("-s","--size", type=int,required=True,help="New images size.")
args = vars(ap.parse_args())

utilMethods = Utils()
utilMethods.resizeImagesInFolder(args['dir'],args['size'])
