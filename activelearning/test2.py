import glob
import os
import pathlib
from ActiveLearning import ActiveLearningClassifier
from DecisionModel import DecisionModel
from Dataset import Dataset
import tensorflow_datasets as tfds
import numpy as np
import cv2

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

model = DecisionModel()
ds = Dataset("D:/4_KULIAH_S2/Summer_Project/summer_project_v2/activelearning/dataset/")
activeAgent = ActiveLearningClassifier(model, ds)

img_dir = pathlib.Path("D:/4_KULIAH_S2/Summer_Project/summer_project_v2/activelearning/dataset/")

# Get a list of all subdirectories
directories = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]

x = []
y = []

for folder_name in directories:
    folder_path = os.path.join(img_dir, folder_name)

    # Get a list of all image files in the subdirectory
    image_files = glob.glob(os.path.join(folder_path, '*.png'))

    # Loop over the image files
    for image_file in image_files:
        # Read the image file and convert it to a numpy array
        image = cv2.imread(image_file)
        x.append(image)
        y.append(folder_name)

x = np.array(x)
y = np.array(y)
split_idx = len(y)

imgs = model.encode_images(x)

# print(imgs.shape)

for i in range(split_idx):
    ds.add_data(imgs[i].reshape(1, -1), np.array([y[i]]))

activeAgent.retrain()

# ds.to_file()
img = cv2.imread('D:/4_KULIAH_S2/Summer_Project/summer_project_v2/activelearning/dataset/0/img1.png')
label = activeAgent.predict(np.array([img]))
print(label)
