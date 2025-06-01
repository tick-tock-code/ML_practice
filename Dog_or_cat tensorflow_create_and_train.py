""" Code to CREATE AND TRAIN an ML model to predict whether a photo is that of a cat or a dog 
    using data from kaggle or google?"""

# two packages
# conda install anaconda::cudnn nvidia::cuda

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import re
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Precision, Recall, AUC
import imghdr
from PIL import Image, UnidentifiedImageError
import random
import sys
import signal
from tensorflow.keras.callbacks import ModelCheckpoint




""" --- Defining functions ---"""
def create_model_ox_cat_dog(input_shape, num_classes):
    """Creates a CNN model for image classification."""
    regulariser_factor = 0.001
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),  # Input layer - comments assume img 128x128x3, but actually 500x500x3

            # Convolutional layer creates 32 3x3 filters, turning image of 128,128,3 to 126,126,32 - conv2d automatically compresses 2nd dimension (rgb - depth) to 1
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(regulariser_factor)), # relu makes sure the output is 0 or positive (max(0,x))

            # Pooling 2D layer with pool size 2x2, reducing the image size to 63,63,32 - doesnt affect depth
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Convolutional layer creates 64 3x3 filters, turning image of 63,63,32 to 61,61,64
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(regulariser_factor)),

            # Pooling 2D layer with pool size 2x2, reducing the image size to 30,30,64
            layers.MaxPooling2D(pool_size=(2, 2)),

            
            # Convolutional layer creates 64 3x3 filters, turning image of 63,63,32 to 61,61,64
            #layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),

            # Pooling 2D layer with pool size 2x2, reducing the image size to 30,30,64
            #layers.MaxPooling2D(pool_size=(2, 2)),
            

            # Flatten layer to flatten the image to 1D array of 30*30*64 = 57600
            layers.Flatten(),

            # Dense layer with 128 neurons
            layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(regulariser_factor)),

            # layers.Dropout(0.5),

            # Dense layer with 2 neurons (2 classes: cat/dog)
            layers.Dense(num_classes, activation="softmax"),  # Output layer (2 classes: cat/dog)
        ]
    )
    return model



""" Unused """
def convert_to_16bit(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB") # Ensure it's in RGB format
    img_16bit = np.array(img, dtype=np.uint16) * 257  # Simple upsampling
    return img_16bit

def normalize_image(image_path):
    img = Image.open(image_path)
    image = np.array(img)
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    else:
        raise ValueError("Unsupported bit depth")




""" Unused Ends"""

def rotate_image(image, angle_in):
    """Rotates the image by a given angle."""
    k = tf.cast(tf.round(angle_in / 90), dtype=tf.int32)
    return tf.image.rot90(image, k=k)

def augment_image(image, label):
    """Augments the image by reflections and rotations."""
    augmented_images = []
    augmented_labels = []

    # Original Image
    augmented_images.append(image)
    augmented_labels.append(label)

    # Reflections
    augmented_images.append(tf.image.flip_left_right(image))  # X-axis reflection
    augmented_labels.append(label)
    augmented_images.append(tf.image.flip_up_down(image))  # Y-axis reflection
    augmented_labels.append(label)
    augmented_images.append(tf.image.flip_left_right(tf.image.flip_up_down(image)))  # Both X and Y
    augmented_labels.append(label)

    # Rotations (for each of the above 4 images)
    rotated_aug_images = []
    rotated__aug_labels = []

    for img, lbl in zip(augmented_images, augmented_labels):
        for angle in [0, 90, 180, 270]:
            rotated_img = rotate_image(img, tf.constant(angle * np.pi / 180, dtype=tf.float32))
            rotated_aug_images.append(rotated_img)
            rotated__aug_labels.append(lbl)

    return rotated_aug_images, rotated__aug_labels

def get_image_paths_and_labels(data_dir):
    """
    Gets image paths and labels from filenames in a directory.

    Args:
        data_dir: Path to the directory containing the images.

    Returns:
        A tuple containing two lists:
        - image_paths: A list of full paths to the image files.
        - labels: A list of corresponding labels.
        Returns empty lists if no suitable files are found.
    """

    image_paths = []
    labels = []

    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for common image extensions
            match = re.match(r"([a-zA-Z]+(?:_[a-zA-Z]+)*)_(\d+)\.(?:png|jpg|jpeg)", filename, re.IGNORECASE) #Use a regex to extract the label
            if match:
                label = match.group(1)

                try:
                    label = label.encode('latin-1').decode('utf-8') #Try decoding with utf-8, if it fails then try latin-1
                except UnicodeDecodeError:
                    label = label.encode('utf-8').decode('latin-1')

                image_path = os.path.join(data_dir, filename)
                image_paths.append(image_path)
                labels.append(label)

    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    image_paths, labels = zip(*combined)

    return image_paths, labels


def load_image(image_path, label, img_height, img_width):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3) # Or decode_png if using png
    image = tf.cast(image, dtype=tf.uint16)  # Convert to uint8
    image = tf.image.resize(image, (img_height, img_width))
    return image, label

def create_dataset(image_paths, labels, img_height, img_width, batch_size):
    """Creates a tf.data.Dataset from image paths and labels."""


    labels_tensor = tf.constant(labels)
    unique_labels_tensor = tf.unique(labels_tensor).y
    
    # Decode byte strings to Unicode strings
    unique_labels_list = [label.decode('utf-8') for label in unique_labels_tensor.numpy()]
    unique_labels = np.array(unique_labels_list)
    num_classes = len(unique_labels)

    print("Unique labels:", unique_labels)

    label_to_index = {label: index for index, label in enumerate(unique_labels)} # dictionary of labels to index
    numeric_labels = [label_to_index[label] for label in labels] # list of indices for each label

    # One-hot encode the numeric labels
    numeric_labels = tf.keras.utils.to_categorical(numeric_labels, num_classes=num_classes) # added num_classes

    image_paths_tensor = tf.convert_to_tensor(image_paths, dtype=tf.string)
    numeric_labels_tensor = tf.convert_to_tensor(numeric_labels)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths_tensor, numeric_labels_tensor))
    dataset = dataset.map(lambda image_path, label: load_image(image_path, label, img_height, img_width)) # variables : function on variables   ,num_parallel_calls=tf.data.AUTOTUNE - removed

    """ --- For Augmentation Only --- """
    dataset = dataset.flat_map(lambda image, label: tf.data.Dataset.from_tensor_slices(augment_image(image, label))) #flat_map to create new dataset


    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    #dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) #Prefetching - cant find autotune??? (AUTOTUNE = tf.data.experimental.AUTOTUNE)
    #dataset = dataset.cache() #Caching - not sure I want to cache the entire dataset in my memory at once - too big
    return dataset, unique_labels, num_classes




def check_image_old(filename):
  try:
    with Image.open(filename) as img:
      img.verify()  # Raises an exception if corrupt
      return True, None
  except (OSError, IOError, Exception) as e:  # Catch various exceptions
    return False, str(e)


def check_image(filename):
    try:
        with Image.open(filename) as img:
            img.load()
            if img.size == (0, 0):
                return False, "Image has zero dimensions"
            if img.format not in ("JPEG"):
                return False, f"Unexpected image format: {img.format}"
            #Check for other unusual properties
            if os.path.getsize(filename) < 100:
                return False, f"File is too small: {os.path.getsize(filename)} bytes"
            return True, None
    except (OSError, SyntaxError, UnidentifiedImageError) as e:
        return False, str(e)
    if os.path.getsize(filename) < 100: #Check for small files
        return False, f"File is too small: {os.path.getsize(filename)} bytes"

    return True, None

def find_corrupted_images(directory):
  """Finds corrupted JPEG images in a directory."""
  corrupted_files = []
  for filename in os.listdir(directory):
    if filename.lower().endswith(('.jpg', '.jpeg')):
      filepath = os.path.join(directory, filename)
      is_valid, error_message = check_image(filepath)
      if not is_valid:
        corrupted_files.append((filepath, error_message))
        print(f"File {filepath} is corrupted: {error_message}")
  return corrupted_files

def remove_corrupted(corrupted):
  if corrupted:
    print(f"Found {len(corrupted)} corrupted images:")
    for file, err in corrupted:
      print(f"- {file}: {err}")
  else:
    print("No corrupted images found in the specified directory.")

  # Optionally remove the corrupted files with enhanced error handling
  if corrupted:
    remove_corrupted = input("Do you want to remove these files? (y/n): ")
    if remove_corrupted.lower() == 'y':
      for file, _ in corrupted:
        try:
          os.remove(file)
          print(f"Removed: {file}")
        except (OSError, PermissionError) as e:
          print(f"Error removing {file}: {e}")
    else:
      print("Corrupted files were not removed.")



def signal_handler(sig, frame):
  print('You pressed Ctrl+C!')
  # Your code to be run on interrupt
  try:
      print("Training interrupted by user. Saving model...")
      model.save(model_path)  # Save the model
      print("Model saved.")
  except Exception as e: # added exception handling
      print(f"Error saving model: {e}")
  finally:
      sys.exit(0)  # Exit cleanly after saving





""" --- 
---
---
---
---
SCRIPT STARTS HERE
---
---
---
---
---
---
 ---"""


""" --- Defining Paths --- """
current_dir = os.path.dirname(os.path.abspath(__file__))
data_directory = os.path.join(current_dir, 'Training_data', 'images')  # Replace with the actual path
model_path = os.path.join(current_dir, "autosave_model.h5")

# Defining callback
checkpoint_callback = ModelCheckpoint(
    filepath=model_path,
    save_freq=100,  # Save every 100 batches
    save_weights_only=False,  # Save the entire model (set to True to save only weights)
    verbose=1  # Print a message when saving the model
)


""" --- Load data and create dataset ---"""
# Find corrupted images - remove them
corrupted = find_corrupted_images(data_directory)
remove_corrupted(corrupted)

# After removing corrupted images, then identify paths/labels
image_paths, labels = get_image_paths_and_labels(data_directory)

# Creating dataset
if image_paths: #check if any images were found
    print(f"Found {len(image_paths)} images.")
    img_height = 500
    img_width = 500
    batch_size = 32
    dataset, unique_labels, num_classes = create_dataset(image_paths, labels, img_height, img_width, batch_size)
    print(unique_labels)


    # Print shapes of a single batch shape = (batch_size, x,y,rgb_depth)
    for images, labels in dataset.take(1):
        print(images.shape)
        print(labels.shape)
else:
    print("No images found in the directory.")






""" --- Creating the model  ---"""
# Defining training and evaluation data
length_dataset = dataset.reduce(0, lambda x, _: x + 1).numpy()
train_size = int(0.8 * length_dataset)
val_size = length_dataset - train_size
train_ds = dataset.take(train_size)
test_ds = dataset.skip(train_size)

# Define input shape
input_shape = (img_height, img_width, 3) # Define the input shape
print('Num classes:', num_classes)

# Create model
model = create_model_ox_cat_dog(input_shape, num_classes)
 
# Compile the model - loss/optimiser define how the model is trained,
"""
initial_learning_rate = 0.001
decay_steps = 10000  # Number of steps to decay over
decay_rate = 0.96  # Decay rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps, decay_rate, staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
"""
optimizer = tf.keras.optimizers.Adam(clipnorm=1, clipvalue=0.5)  # Clip gradients with norm above 1.0
model.compile(loss="categorical_crossentropy", 
    optimizer=optimizer,
    metrics=["accuracy", Precision(), Recall(), AUC()]) # Metrics to record
model.summary() # Print model summary


""" --- Training Model --- """
# Train the model
epochs = 10  # Number of training epochs

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=epochs,
    callbacks=[checkpoint_callback]
)


# Saving model if training successful
model.save(model_path)
print(f"Model saved to: {model_path}") #Print the full path



""" --- Plotting the training history --- """
metrics_to_plot = ['accuracy', 'precision', 'recall']  # Metrics to plot - loss left out
num_metrics = len(metrics_to_plot)

plt.figure(figsize=(15, 5 * num_metrics))  # Adjust figure size for subplots

for i, metric_name in enumerate(metrics_to_plot):
    plt.subplot(num_metrics, 1, i + 1)  # Create subplots in a vertical layout

    metric_values = history.history[metric_name]
    plt.plot(metric_values, label=f'Training {metric_name.capitalize()}')

    if f'val_{metric_name}' in history.history: #check if validation data exists
        val_metric_values = history.history[f'val_{metric_name}']
        plt.plot(val_metric_values, label=f'Validation {metric_name.capitalize()}')

    plt.title(f'Model {metric_name.capitalize()}')
    plt.ylabel(metric_name.capitalize())
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Save metric values as NumPy arrays
for metric_name in history.history:
    metric_values = np.array(history.history[metric_name])
    np.save(f"{metric_name}_dogcat.npy", metric_values)
    print(f"saved {metric_name}")