import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from extra_keras_datasets import emnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers


policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.4,
    zoom_range=0.4,
    fill_mode='nearest',
)

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = emnist.load_data(type='byclass')

# Preprocess the data
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
test_images = test_images.astype("float32") / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# Define a simple neural network architecture
model = Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1), padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.001), padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.001),padding='same' ),
        layers.MaxPooling2D((2, 2), padding='same'),

        layers.Conv2D(512, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.001), padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(2048, activation="relu", kernel_regularizer=regularizers.l2(0.001)),

        layers.Dense(1024, activation="relu", kernel_regularizer=regularizers.l2(0.001)),


        layers.Dropout(0.5),
        layers.Dense(62, activation="softmax")
])


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train_generator = datagen.flow(train_images, train_labels, batch_size=128)
# Train the model on the dataset

early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

model.fit(train_generator, epochs=100, validation_data=(test_images, test_labels), callbacks=[early_stopping], verbose=1)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc:.2f}')

model.save("mnist_digit_recognition_model.h5")