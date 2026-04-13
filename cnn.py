import os
import random
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import argparse
from sklearn.model_selection import train_test_split
import numpy as np


def parse_arguments():
    """Parses command-line arguments for training the CNN."""
    parser = argparse.ArgumentParser(description="Train a CNN for bed bug classification.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training.")
    parser.add_argument('--im_height', type=int, default=256, help="Height of input images.")
    parser.add_argument('--im_width', type=int, default=256, help="Width of input images.")
    parser.add_argument('--path', type=str, help="Path to the dataset directory.")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument('--val_size', type=float, default=0.2, help="Proportion of the dataset to include in the validation split.")
    parser.add_argument('--random_state', type=int, default=1234, help="Random state for reproducibility.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train the model.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer.")
    return parser.parse_args()

def load_image(image_path, label, imHeight, imWid):
    """Reads an image from a file, decides it, resizes it, and normalizes the pixel values."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [imHeight, imWid])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
    return image, label

def get_paths_labels(X, metadata, inpath):
    """Gets the image paths and labels for a given set of subjects."""
    image_paths = []
    image_labels = []
    for subject in X:
        paths = metadata[metadata['subject'] == subject]['Filename'].tolist()
        paths = [os.path.join(inpath, path) for path in paths]
        labels = metadata[metadata['subject'] == subject]['species'].tolist()
        image_paths.extend(paths)
        image_labels.extend(labels)
    return image_paths, image_labels



def encode_labels(labels):
    """Encodes string labels into integer vectors."""
    labels = np.array(labels)
    encoded = (labels == "lectularius").astype(int)
    return encoded

def prepare_data(path, test_size, val_size, random_state, imHeight, imWid, inpath):
    """Prepares the training, validation, and test datasets."""

    # Load metadata
    metadata = pd.read_csv(os.path.join(path, "allMetadata.csv"))
    

    # get unique subjects and split them into training, validation, and test sets, keeping class numbers equal. 
    hemipterus = metadata[metadata['species'] == 'hemipterus']['subject'].unique()
    lectularius = metadata[metadata['species'] == 'lectularius']['subject'].unique()
    labels = ['hemipterus'] * len(hemipterus) + ['lectularius'] * len(lectularius)
    subjects = hemipterus.tolist() + lectularius.tolist()
    
    X_train, X_temp, y_train, y_temp = train_test_split(subjects, labels, test_size=test_size + val_size, stratify = labels, random_state = random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(test_size / (test_size + val_size )), stratify = y_temp, random_state = random_state)
    print(f"Training subjects: {len(X_train)}"
          f"\nValidation subjects: {len(X_val)}"
          f"\nTest subjects: {len(X_test)}")

    # get filenames and labels for training, validation, and test sets
    train_image_paths = []
    train_image_labels = []
    train_image_paths, train_image_labels = get_paths_labels(X_train, metadata, inpath)
    val_image_paths, val_image_labels = get_paths_labels(X_val, metadata, inpath)
    test_image_paths, test_image_labels = get_paths_labels(X_test, metadata, inpath)
    with open("data_split.txt", "w") as f:
        f.write(f"Training subjects: {X_train}\n")
        f.write(f"Validation subjects: {X_val}\n")
        f.write(f"Test subjects: {X_test}\n")

    # Encode labels
    train_image_labels = encode_labels(train_image_labels)
    val_image_labels = encode_labels(val_image_labels)
    test_image_labels = encode_labels(test_image_labels)

    # creeate data
    training_data = tf.data.Dataset.from_tensor_slices((train_image_paths, train_image_labels))
    validation_data = tf.data.Dataset.from_tensor_slices((val_image_paths, val_image_labels))
    test_data = tf.data.Dataset.from_tensor_slices((test_image_paths, test_image_labels))

    training_data = training_data.map(lambda x, y: load_image(x, y, imHeight, imWid), num_parallel_calls=tf.data.AUTOTUNE)
    validation_data = validation_data.map(lambda x, y: load_image(x, y, imHeight, imWid), num_parallel_calls=tf.data.AUTOTUNE)

    return(training_data, validation_data, test_data)

def build_cnn(imHeight, imWid):
    model = models.Sequential([
        layers.Conv2D(45, (3, 3), activation='relu',
                      input_shape=(imHeight, imWid, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(30, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def fit_cnn(model, training_data, validation_data, epochs, batch_size, learning_rate):
    training_data = training_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    validation_data = validation_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    history = model.fit(training_data, epochs=epochs, validation_data=validation_data)
    return history

def main():

    # Parse command-line arguments.
    args = parse_arguments()

    # Divide data into training, validation, and test sets.
    training_data, validation_data, test_data = prepare_data(args.path, args.test_size, args.val_size, args.random_state, args.im_height, args.im_width, args.path)

    # build CNN
    model = build_cnn(args.im_height, args.im_width)

    # train CNN
    history = fit_cnn(model, training_data, validation_data, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)

if __name__ == "__main__":
    main()
