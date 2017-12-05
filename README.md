# cnn_fire
Convolutional neural network to detect fire

## Scripts

* cnn_architectures.py - Defines several CNN architectures;
* fcmmanager.py - Sends push notifications to firebase service
* googlemanager.py - Manages the connection with goodle drive service
* nn_test.py - Class to handle testing one image or video in one previously recorded CNN train
* raw_images_loader.py - Load raw images into numpy arrays to be consumed by CNN
* test_on_cam.py - Apply treined CNN on image captured by camera
* train_manual.py - Train the network manually. Read from raw images, augmenting manualy etc
* train.py - Train network with Keras generator
* confusion_matrix.py - Plot the confusion matrix for a given trained network
* plot_filtesrs.py - Plot treined network learned filters
* resizeImagesInFolder.py - Resize images in one folder
* renameimagesindir.py - Rename the images in one dir so that we can distinguish between them on training time
* images_downloader.py - Download images from some services on internet
* deleteDuplicateImages.py - Deletes duplicate images in a given folder

## Files
* model.json - The stores json architecture from training with its parameters setted.
* wights.h5 - The network treined weights

## Packages Reqs:
- Keras
- TensorFlow
- Numpy
- Matplotlib
- Scipy
- ApiClient
- Imutils
- OpenCv
- Argparse
- MoviePy
- SimpleJson
- PyFcm
- Oauth2client
- Httplib2
