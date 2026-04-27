import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from utils import load_image, parse_val_csvs, encode_labels, prepare_data, get_weights, batch_and_augment

def parse_arguments():
    """Parses command-line arguments for training the CNN."""
    parser = argparse.ArgumentParser(description="Train a CNN for bed bug classification.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train the model.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--im_height', type=int, default=256, help="Height of input images.")
    parser.add_argument('--im_width', type=int, default=256, help="Width of input images.")
    parser.add_argument('--train_csv', type=str, help="Path to the training data.")
    parser.add_argument( "--val_csvs", nargs="*", default=[], help="Validation sets as name=path (e.g. sophie=path.csv inat=path.csv)" )
    parser.add_argument('--save_path', type=str, default="cnn_model.keras", help="Path to save the trained model.")
    parser.add_argument('--augment', action='store_true', help="Whether to apply data augmentation during training.")
    parser.add_argument('--balance', action='store_true', help="Whether to balance classes during training.")
    return parser.parse_args()

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

def fit_cnn(model, training_data, epochs, batch_size, learning_rate, balance=None):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    if balance is not None:
        history = model.fit(training_data, epochs=epochs, class_weight=balance)
    else:
        history = model.fit(training_data, epochs=epochs)

    return history

def evaluate_cnn(model, val_data):
    val_data = val_data.batch(64).prefetch(tf.data.AUTOTUNE)
    results = model.evaluate(val_data)
    print(f"Validation loss: {results[0]}, Validation accuracy: {results[1]}")

def main():
    args = parse_arguments()
    training_data = prepare_data(args.train_csv, args.im_height, args.im_width)
    val_data_dict = {name: prepare_data(path, args.im_height, args.im_width) for name, path in parse_val_csvs(args.val_csvs).items()}

    # compute class weights
    if args.balance:
        class_weights = get_weights(args.train_csv)
        print(class_weights)
    else:
        class_weights = None

    # batching and augmentation
    training_data = batch_and_augment(training_data, args.batch_size, augment=args.augment)

    # build CNN
    model = build_cnn(args.im_height, args.im_width)

    # train CNN
    history = fit_cnn(model, training_data, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, balance=class_weights)

    # validate CNN
    for name, val_data in val_data_dict.items():
        print(f"Evaluating on {name} validation set:")
        evaluate_cnn(model, val_data)

    # save model
    model.save(args.save_path)


if __name__ == "__main__":
    main()