import numpy as np
import cv2
from tensorflow.keras.models import load_model

def preprocess_image(image_path):
    # Load the image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to 28x28 pixels
    img = cv2.resize(img, (28, 28))

    # Invert the image (MNIST images are white digits on a black background)

    # Normalize the pixel values to the range [0, 1]
    img = img / 255.0

    # Expand the dimensions of the image to match the input shape (1, 28, 28)
    img = np.expand_dims(img, axis=0)

    return img

def predict_digit(image_path, model):
    # Preprocess the input image
    img = preprocess_image(image_path)
    
    # Make a prediction using the model
    prediction = model.predict(img)

    # Get the digit with the highest probability
    #print(prediction)
    digit = np.argmax(prediction)

    return digit

# Load the saved model
model = load_model('mnist_digit_recognition_model.h5')


# Path to the image file containing the handwritten digit
image_path = 'image.jpg'


# Predict the digit using the model
predicted_digit = predict_digit(image_path, model)


print(f'The predicted digit is: {predicted_digit}')
