from utils import Utils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dir",required=True,help="Dir where images are.")
ap.add_argument("-b","--batchsize",type=int,required=True,help="Batches to be augmented")
ap.add_argument("-f","--finaldir",required=True,help="Dir to save images augmented")
args = vars(ap.parse_args())

utilMethods = Utils()
utilMethods.augmentImagesInFolder(args['dir'],args['batchsize'],args['finaldir'],for_validation_dataset=True)
