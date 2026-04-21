import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def parse_arguments():
    """Parses command-line arguments for training the CNN."""
    parser = argparse.ArgumentParser(description="Train a CNN for bed bug classification.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train the model.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--im_height', type=int, default=256, help="Height of input images.")
    parser.add_argument('--im_width', type=int, default=256, help="Width of input images.")
    parser.add_argument('--train_csv', type=str, help="Path to the training data.")
    parser.add_argument('--val_csvs', nargs="*", default=[], help="Validation sets as name=path (e.g. sophie=path.csv inat=path.csv)")
    parser.add_argument('--finetune_csv', type=str, help="Path to the finetuning data.")
    parser.add_argument('--finetune_epochs', type=int, default=5, help="Epochs for fine-tuning.")
    parser.add_argument('--finetune_lr', type=float, default=1e-4, help="Learning rate for fine-tuning.")
    parser.add_argument('--save_path', type=str, default="finetune_model.keras", help="Path to save the trained model.")
    return parser.parse_args()

def load_image(image_path, label, imHeight, imWid):
    """Reads an image from a file, decides it, resizes it, and normalizes the pixel values."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [imHeight, imWid])
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

def prepare_data(csv, imHeight, imWid):
    raw_data = pd.read_csv(csv)
    image_paths = raw_data["Path"].tolist()
    labels = raw_data["species"].tolist()
    encoded_labels = encode_labels(labels)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, encoded_labels))
    dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)
    dataset = dataset.map(lambda x, y: load_image(x, y, imHeight, imWid), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return dataset

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

def fit_cnn(model, training_data, epochs, batch_size, learning_rate):
    training_data = training_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    history = model.fit(training_data, epochs=epochs)

    return history

def finetune_cnn(model, finetune_data, epochs, batch_size, learning_rate, freeze_layers=True):
    finetune_data = finetune_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if freeze_layers:
        for layer in model.layers[:-2]:  # freeze all but last 2 layers
            layer.trainable = False

    # Recompile with lower LR
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\nStarting fine-tuning...")
    history = model.fit(finetune_data, epochs=epochs)

    return history

def evaluate_cnn(model, val_data):
    val_data = val_data.batch(64).prefetch(tf.data.AUTOTUNE)
    results = model.evaluate(val_data)
    print(f"Validation loss: {results[0]}, Validation accuracy: {results[1]}")

def main():
    args = parse_arguments()
    training_data = prepare_data(args.train_csv, args.im_height, args.im_width)
    finetune_data = prepare_data(args.finetune_csv, args.im_height, args.im_width)
    val_data_dict = {name: prepare_data(path, args.im_height, args.im_width) for name, path in parse_val_csvs(args.val_csvs).items()}

    # build CNN
    model = build_cnn(args.im_height, args.im_width)

    # train CNN
    history = fit_cnn(model, training_data, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)

    # finetune CNN
    finetune_cnn(model, finetune_data, epochs=args.finetune_epochs, batch_size=args.batch_size, learning_rate=args.finetune_lr, freeze_layers=True)

    # validate CNN
    for name, val_data in val_data_dict.items():
        print(f"Evaluating on {name} validation set:")
        evaluate_cnn(model, val_data)

    # save model
    model.save(args.save_path)

if __name__ == "__main__":
    main()