from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
import argparse

"""

Script to make augmented images

augment_image --image images/image_to_be_augmented.jpg

"""

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	help="path to the input image")
args = vars(ap.parse_args())


datagen = ImageDataGenerator(
	rotation_range = 40,
	width_shift_range = 0.2,
	height_shift_range = 0.2,
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest'
)

img = load_img('args["image"]')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0

for batch in datagen.flow(x,batch_size=1,
	save_to_dir='./',
	save_prefix='aug', 
	save_format='jpg'):

	i += 1
	if i > 20:
		break
