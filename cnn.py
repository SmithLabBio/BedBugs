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
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--im_height', type=int, default=256, help="Height of input images.")
    parser.add_argument('--im_width', type=int, default=256, help="Width of input images.")
    parser.add_argument('--path', type=str, help="Path to the metadata file.")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument('--val_size', type=float, default=0.2, help="Proportion of the dataset to include in the validation split.")
    parser.add_argument('--random_state', type=int, default=1234, help="Random state for reproducibility.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train the model.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--im_path", type=str, help="Path to the directory containing the images.")
    parser.add_argument("--train_sources", nargs="+", choices=["inat", "sophie"], default=["inat", "sophie"], help="Sources to include in the training set.")
    return parser.parse_args()

def load_image(image_path, label, imHeight, imWid):
    """Reads an image from a file, decides it, resizes it, and normalizes the pixel values."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [imHeight, imWid])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
    return image, label

def encode_labels(labels):
    """Encodes string labels into integer vectors."""
    labels = np.array(labels)
    encoded = (labels == "lectularius").astype(int)
    return encoded

def split_subjects(metadata, test_size, val_size, random_state):
    """Split a metadata subset into train/val/test at subject level."""

    hem = metadata[metadata['species'] == 'hemipterus']['subject'].unique()
    lec = metadata[metadata['species'] == 'lectularius']['subject'].unique()

    labels = ['hemipterus'] * len(hem) + ['lectularius'] * len(lec)
    subjects = hem.tolist() + lec.tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        subjects, labels,
        test_size=test_size + val_size,
        stratify=labels,
        random_state=random_state
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(test_size / (test_size + val_size)),
        stratify=y_temp,
        random_state=random_state
    )

    return X_train, X_val, X_test

def prepare_data(path, test_size, val_size, random_state,
                 imHeight, imWid, inpath,
                 train_sources):

    metadata = pd.read_csv(inpath)

    metadata_sophie = metadata[metadata['source'] == 'Sophie']
    metadata_inat   = metadata[metadata['source'] == 'iNaturalist']

    # --- Split each source independently ---
    s_train, s_val, s_test = split_subjects(
        metadata_sophie, test_size, val_size, random_state
    )

    i_train, i_val, i_test = split_subjects(
        metadata_inat, test_size, val_size, random_state
    )

    train_sources = set(train_sources)

    # --- Apply your rules ---
    if train_sources == {"inat"}:
        X_train = i_train
        
    elif train_sources == {"sophie"}:
        X_train = s_train

    elif train_sources == {"inat", "sophie"}:
        X_train = s_train + i_train


    # --- Convert to paths/labels ---
    train_image_paths, train_image_labels = get_paths_labels(X_train, metadata, path)
    val_image_paths_i, val_image_labels_i = get_paths_labels(i_val, metadata, path)
    val_image_paths_s, val_image_labels_s = get_paths_labels(s_val, metadata, path)
    test_image_paths_i, test_image_labels_i   = get_paths_labels(i_test, metadata, path)
    test_image_paths_s, test_image_labels_s   = get_paths_labels(s_test, metadata, path)

    # Encode
    train_image_labels = encode_labels(train_image_labels)
    val_image_labels_i = encode_labels(val_image_labels_i)
    val_image_labels_s = encode_labels(val_image_labels_s)
    test_image_labels_i  = encode_labels(test_image_labels_i)
    test_image_labels_s  = encode_labels(test_image_labels_s)


    # Datasets
    training_data = tf.data.Dataset.from_tensor_slices((train_image_paths, train_image_labels))
    val_data_i = tf.data.Dataset.from_tensor_slices((val_image_paths_i, val_image_labels_i))
    val_data_s = tf.data.Dataset.from_tensor_slices((val_image_paths_s, val_image_labels_s))
    test_data_i = tf.data.Dataset.from_tensor_slices((test_image_paths_i, test_image_labels_i))
    test_data_s = tf.data.Dataset.from_tensor_slices((test_image_paths_s, test_image_labels_s))

    training_data = training_data.map(lambda x, y: load_image(x, y, imHeight, imWid),
                                      num_parallel_calls=tf.data.AUTOTUNE)
    val_data_i = val_data_i.map(lambda x, y: load_image(x, y, imHeight, imWid),
                               num_parallel_calls=tf.data.AUTOTUNE)
    val_data_s = val_data_s.map(lambda x, y: load_image(x, y, imHeight, imWid),
                               num_parallel_calls=tf.data.AUTOTUNE)

    return training_data, val_data_i, val_data_s, test_data_i, test_data_s

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

def fit_cnn(model, training_data, val_data_i, val_data_s, epochs, batch_size, learning_rate):
    training_data = training_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_data_i = val_data_i.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_data_s = val_data_s.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    history = model.fit(training_data, epochs=epochs)
    
    print("Sophie val:")
    model.evaluate(val_data_s)

    print("iNat val:")
    model.evaluate(val_data_i)
    return history

def main():

    # Parse command-line arguments.
    args = parse_arguments()

    # Divide data into training, validation, and test sets.
    training_data, val_data_i, val_data_s, test_data_i, test_data_s = prepare_data(args.im_path, args.test_size, args.val_size, args.random_state, args.im_height, args.im_width, args.path, args.train_sources)

    # build CNN
    model = build_cnn(args.im_height, args.im_width)

    # train CNN
    history = fit_cnn(model, training_data, val_data_i, val_data_s, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)

if __name__ == "__main__":
    main()
