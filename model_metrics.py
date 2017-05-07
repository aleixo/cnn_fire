from nn_test import NnTest
from raw_images_loader import RawImagesLoader
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d","--imagesdir",required=True,help="Where images are")
args = vars(ap.parse_args())

images_path = args["imagesdir"]
classifier = NnTest()
images_loader = RawImagesLoader()

X_train,Y_train,X_test,Y_test,X_val,Y_val = images_loader.getImagesRepresentation(
    images_path,
    64,
    2,
    forRGB=True,
    test_size=0.2,
    val_size=0.3
    )

model = classifier.loaded_model
score = model.evaluate(X_val,Y_val)
print(score)
