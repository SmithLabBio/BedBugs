import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.utils.class_weight import compute_class_weight

def load_image(image_path, label, imHeight, imWid, pretrain=False):
    """Reads an image from a file, decides it, resizes it, and normalizes the pixel values."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [imHeight, imWid])
    if pretrain:
        image = tf.cast(image, tf.float32)
    else:
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
    return image, label

def parse_val_csvs(val_list): 
    val_dict = {}
    for item in val_list:
        if "=" not in item: 
            raise ValueError(f"Invalid val_csv format: {item}. Use name=path")
        name, path = item.split("=", 1)
        val_dict[name] = path
    return val_dict

def encode_labels(labels):
    """Encodes string labels into integer vectors."""
    labels = np.array(labels)
    encoded = (labels == "lectularius").astype(int)
    return encoded

def prepare_data(csv, imHeight, imWid, pretrain=False):
    raw_data = pd.read_csv(csv)
    image_paths = raw_data["Path"].tolist()
    labels = raw_data["species"].tolist()
    encoded_labels = encode_labels(labels)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, encoded_labels))
    dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)
    dataset = dataset.map(lambda x, y: load_image(x, y, imHeight, imWid, pretrain), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return dataset

def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_contrast(image, 0.9, 1.1)

    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k)

    return image, label

def batch_and_augment(training_data, batch_size, augment=False):
    if augment:
        print("Applying data augmentation to training data.")
        training_data = (
            training_data
            .map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
            .map(lambda x, y: (preprocess_input(x), y),
                 num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))
    else:
        training_data = (
            training_data
            .map(lambda x, y: (preprocess_input(x), y),
                 num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
    return training_data

def get_weights(training_data):
    raw_df = pd.read_csv(training_data)
    labels = encode_labels(raw_df["species"].tolist())

    classes = np.unique(labels)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    class_weights = dict(zip(classes, class_weights))
    print("Class weights:", class_weights)
    return class_weights