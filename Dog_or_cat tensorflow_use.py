""" Code to CREATE AND TRAIN an ML model to predict whether a photo is that of a cat or a dog 
    using data from kaggle or google?"""


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator



""" --- Defining functions ---"""



""" --- Loading Model and Making Predictions --- """
# Example of loading the model
loaded_model = keras.models.load_model("cat_dog_model")

# Example of making predictions
predictions = loaded_model.predict(x_train)
print(predictions)

