#!/bin/sh
python augmentForImagesInFolder.py -d ~/Desktop/Fire/ -b 16 -f ~/Desktop/Fire/
python deleteDuplicateImages.py -d ~/Desktop/Fire/
python resizeImagesInFolder.py -d ~/Desktop/Fire/ -s 64
python renameImagesInDir.py -d ~/Desktop/Fire/ -p fire
python convertImagesInDirToGrayscale.py -d ~/Desktop/ForrestAug/
