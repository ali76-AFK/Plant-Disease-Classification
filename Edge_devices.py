# Download dataset
# !pip install -q gdown

# plant_leave_diseases_train.zip
# !gdown https://drive.google.com/uc?id=1MCQ2ldiKZUeVM1rVw1gPlBaX43AJB3R0

# plant_leave_diseases_test.zip/ install these files either on your central device or in google colab
# !gdown https://drive.google.com/uc?id=1yqvfEVeb0IAutxZK83_wUoUWm5apYSF8

# import zipfile
#
# # Unzip data
# with zipfile.ZipFile('plant_leave_diseases_train.zip', 'r') as zip_file:
#     zip_file.extractall()
#
# with zipfile.ZipFile('plant_leave_diseases_test.zip', 'r') as zip_file:
#     zip_file.extractall()

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Paths to your folders
train_data_dir = 'plant_leave_diseases_train'

# Image settings
img_size = (256, 256)
batch_size = 32

# Create a ImageDataGenerator with validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training data generator
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Validation data generator
val_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

model = Sequential([
    Input(shape=(img_size[0], img_size[1], 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same'), # Added another layer
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'), # Increased units
    Dropout(0.5), # Added dropout
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Increased epochs
model.fit(train_generator, validation_data=val_generator, epochs=15)

# Convert the Keras model to a TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('plant_disease_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite model saved to plant_disease_model.tflite")


import os
import numpy as np
import pandas as pd
from PIL import Image

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="plant_disease_model_3.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare the test data
test_data_dir = 'plant_leave_diseases_test'
test_image_paths = []
for filename in os.listdir(test_data_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        test_image_paths.append(os.path.join(test_data_dir, filename))

# Get the class names from the training generator (assuming the same class order)
class_names = list(train_generator.class_indices.keys())

results = []

# Iterate through test images and make predictions
for image_path in test_image_paths:
    # Load and preprocess the image
    img = Image.open(image_path).resize(img_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1]

    # Set the tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)

    # Get the predicted class name
    predicted_class = class_names[prediction]

    # Extract the image ID (filename without extension)
    image_id = os.path.splitext(os.path.basename(image_path))[0]

    results.append({'ID': image_id, 'class': predicted_class})

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_df.to_csv('predictions.csv', index=False)

print("Predictions saved to predictions.csv")
