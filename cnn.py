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


# Step 2a: Load and preprocess the training dataset
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


# Step 2b: Load and preprocess the Validation dataset
val_image_paths = []
val_labels = []

for i in range(40, 48):

    val_image_paths.append(
        metaData.loc[metaData['subject'] == i, ["Filename"]])
    val_labels.append(metaData.loc[lambda df: df['subject'] == i, ["species"]])


for i in range(8, 16):
    val_image_paths.append(
        metaData.loc[lambda df: df['subject'] == i, ["Filename"]])
    val_labels.append(metaData.loc[lambda df: df['subject'] == i, ["species"]])

val_image_paths = pd.concat(val_image_paths)
val_labels = pd.concat(val_labels)

# Shuffles validation set
indMapping = [i for i in range(len(val_labels))]
random.shuffle(indMapping)
val_image_paths = val_image_paths.iloc[indMapping]
pd.set_option('future.no_silent_downcasting', True)
valOneHot = val_labels.replace("lectularius", 0)
valOneHot = valOneHot.replace("hemipterus", 1)

# convert pd dataframe into a list
valOneHot = list(valOneHot.to_numpy().flatten())
val_image_paths = list(val_image_paths.to_numpy().flatten())

# Create TensorFlow Dataset for validation data
valDf = tf.data.Dataset.from_tensor_slices((val_image_paths, valOneHot))
valDf = valDf.map(preprocess_image)
valDf = valDf.shuffle(buffer_size=len(val_image_paths)).batch(batch_size)


# Step 3: Display some example images
os.chdir(picPath)

'''for images, labels in trainDf.take(1):  # Take 1 batch as an example
    plt.figure(figsize=(10, 10))
    for i in range(3):  # Display 3 example images
        plt.imshow(images[i])
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.show()'''


# Step 4: Define the CNN model
model = models.Sequential([
    layers.Conv2D(45, (3, 3), activation='relu',
                  input_shape=(imHeight, imWid, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(30, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='sigmoid'),
    layers.Dense(nClass)

])

# Step 5: Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])
model.summary()
# Step 6: Train the model with validation data
history = model.fit(trainDf, epochs=7, validation_data=valDf)

# Step 7: Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
plt.legend()
plt.show()
