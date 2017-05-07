#!/bin/sh
python train.py -a custom > terminal_output.txt
python train.py -a vgg >> terminal_output.txt
python train.py -a karpathy >> terminal_output.txt
python train.py -a lenet >> terminal_output.txt