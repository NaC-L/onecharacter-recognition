import numpy as np
import cv2
from tensorflow.keras.models import load_model

from extra_keras_datasets import emnist

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.4,
    zoom_range=0.4,
    fill_mode='nearest',
)
(train_images, train_labels), (test_images, test_labels) = emnist.load_data(type='byclass')

train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
test_images = test_images.astype("float32") / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model = load_model('mnist_digit_recognition_model.h5')

train_generator = datagen.flow(train_images, train_labels, batch_size=128)
model.fit(train_generator, epochs=100, validation_data=(test_images, test_labels), verbose=1)



model.save("emnist_digit_recognition_model.h5")