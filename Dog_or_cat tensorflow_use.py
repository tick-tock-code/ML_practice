""" Applies trained model to make predictions on new images."""

# --- Importing Libraries ---
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image


""" --- Defining functions ---"""






# Defining Params
MODEL_PATH = "autosave_model.h5"
IMG_PATH = "test_image.jpg"   # put test images here, filenames must contain labels or be in subfolders
IMG_SIZE = (500, 500)           # set to model input



""" --- Loading Model and Making Predictions --- """
# Example of loading the model
loaded_model = keras.models.load_model(MODEL_PATH)

# Image preprocessing
img = Image.open(IMG_PATH).convert("RGB")
img = img.resize(IMG_SIZE)
arr = np.array(img).astype("float32") / 255.0    # IMPORTANT: float32, normalized
arr = np.expand_dims(arr, axis=0)  # (1, H, W, C)

# Making predictions
pred = loaded_model.predict(arr)  # shape (1, num_classes)
print(pred)
pred = pred[0]
pred_class_index = np.argmax(pred)
pred_confidence = float(pred[pred_class_index])

classes = ['boxer', 'american_pit_bull_terrier', 'Abyssinian', 'keeshond',
 'english_cocker_spaniel', 'British_Shorthair', 'pug', 'Ragdoll', 'chihuahua',
 'wheaten_terrier', 'leonberger', 'staffordshire_bull_terrier', 'samoyed',
 'newfoundland', 'beagle', 'Siamese', 'japanese_chin', 'Birman', 'basset_hound',
 'Maine_Coon', 'saint_bernard', 'pomeranian', 'Egyptian_Mau', 'Bengal',
 'Sphynx', 'scottish_terrier', 'yorkshire_terrier', 'Russian_Blue',
 'shiba_inu', 'german_shorthaired', 'Persian', 'english_setter' , 
 'great_pyrenees', 'Bombay', 'miniature_pinscher', 'american_bulldog']

print(f"Predicted: {classes[pred_class_index]} ({pred_confidence:.3f})")