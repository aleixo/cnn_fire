from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
import simplejson
from keras.utils import plot_model
import matplotlib.pyplot as plt

save_weights_filename = "more_images.h5"
save_model_filename = "model_more_images.json"
train_images_path = "images/train"
validate_images_path = "images/validate"

batch_size = 16
epochs = 10

K.set_image_dim_ordering('th')

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_images_path,  
        target_size=(32, 32),  
        batch_size=batch_size,
        class_mode='binary')  


validation_generator = test_datagen.flow_from_directory(
        validate_images_path,
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='binary')

history = model.fit_generator(
        train_generator,
        steps_per_epoch=1000,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=200
)

model_json = model.to_json()

with open(save_model_filename, "w") as json_file:
    json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))

model.save_weights(save_weights_filename)

plot_model(model, to_file='model.png')
print(history.history.keys())

def plotDebug():

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

plotDebug()
