import os
import random
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


# Step 1: Define parameters
batch_size = 4
imHeight = 256
imWid = 256
nClass = 2

# Step 2: Define data preprocessing function


def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [imHeight, imWid])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
    return image, label


# Step 3: Load and preprocess the training dataset
picPath = "/Users/sophiemaedo/work/bed_bug_photos"
metaData = []
# load in metadata
metaData = pd.read_csv(picPath+"/image_metadata.csv")
# print(metaData["species"])
train_image_paths = []
train_labels = []


tLectStartI = []
tHemiStartI = []


# finds path for lect training subjecta
for i in range(34, 40):
    train_image_paths.append(
        metaData.loc[metaData['subject'] == i, ["Filename"]])
    train_labels.append(
        metaData.loc[lambda df: df['subject'] == i, ["species"]])

# finds path for hemi training subjecta
for i in range(1, 8):
    train_image_paths.append(
        metaData.loc[lambda df: df['subject'] == i, ["Filename"]])
    train_labels.append(
        metaData.loc[lambda df: df['subject'] == i, ["species"]])

random.seed(42)

train_image_paths = pd.concat(train_image_paths)


train_labels = pd.concat(train_labels)
# Shuffles training set
indMapping = [i for i in range(len(train_labels))]
random.shuffle(indMapping)
train_image_paths = train_image_paths.iloc[indMapping]
pd.set_option('future.no_silent_downcasting', True)
trainOneHot = train_labels.replace("lectularius", 0)
trainOneHot = trainOneHot.replace("hemipterus", 1)
# convert pd dataframe into a list
trainOneHot = list(trainOneHot.to_numpy().flatten())
train_image_paths = list(train_image_paths.to_numpy().flatten())

trainDf = tf.data.Dataset.from_tensor_slices((train_image_paths, trainOneHot))
trainDf = trainDf.map(preprocess_image)
trainDf = trainDf.shuffle(buffer_size=len(train_image_paths)).batch(batch_size)

# Step 3: Display some example images

os.chdir(picPath)
cwd = os.getcwd()
print("Current Working Directory:", cwd)
for images, labels in trainDf.take(1):  # Take 1 batch as an example
    plt.figure(figsize=(10, 10))
    for i in range(3):  # Display 3 example images
        plt.imshow(images[i])
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.show()
