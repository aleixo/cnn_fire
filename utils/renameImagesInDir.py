from utils import Utils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dir",required=True,help="Dir where images are.")
ap.add_argument("-p","--prefix",required=True,help="Image will have the prefix after renamed.")
args = vars(ap.parse_args())

utilMethods = Utils()
utilMethods.renameFilesInFolder(args['dir'],args['prefix'])