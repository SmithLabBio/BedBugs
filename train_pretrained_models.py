import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications import ResNet50

from utils import load_image, parse_val_csvs, encode_labels, prepare_data, batch_and_augment, get_weights

def parse_arguments():
    """Parses command-line arguments for training the CNN."""
    parser = argparse.ArgumentParser(description="Train a CNN for bed bug classification.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train the model.")
    parser.add_argument('--learning_rate', type=float, default=0.00001, help="Learning rate for the optimizer.")
    parser.add_argument('--im_height', type=int, default=256, help="Height of input images.")
    parser.add_argument('--im_width', type=int, default=256, help="Width of input images.")
    parser.add_argument('--train_csv', type=str, help="Path to the training data.")
    parser.add_argument( "--val_csvs", nargs="*", default=[], help="Validation sets as name=path (e.g. sophie=path.csv inat=path.csv)" )
    parser.add_argument('--save_path', type=str, default="cnn_model.keras", help="Path to save the trained model.")
    parser.add_argument('--finetune_epochs', type=int, default=10, help="Number of epochs to fine-tune the model.")
    parser.add_argument('--finetune_lr', type=float, default=1e-5, help="Learning rate for fine-tuning.")
    parser.add_argument('--augment', action='store_true', help="Whether to apply data augmentation during training.")
    parser.add_argument('--balance', action='store_true', help="Whether to balance classes during training.")
    return parser.parse_args()


def fit_pretrained_cnn(model, training_data, epochs, batch_size, learning_rate, augment=False, balance=None):

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    if balance is not None:
        history = model.fit(training_data, epochs=epochs, class_weight=balance)
    else:
        history = model.fit(training_data, epochs=epochs)
    return history


def build_pretrained_cnn(imHeight, imWid, trainable=False):
    # Load pretrained ResNet (exclude top classifier)
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(imHeight, imWid, 3)
    )
    
    # Freeze or unfreeze base model
    base_model.trainable = trainable

    # Build classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    
    return model

def evaluate_cnn(model, val_data):
    val_data = (
        val_data
        .map(lambda x, y: (preprocess_input(x), y),
             num_parallel_calls=tf.data.AUTOTUNE)
        .batch(64)
        .prefetch(tf.data.AUTOTUNE)
    )
    results = model.evaluate(val_data)
    print(f"Validation loss: {results[0]}, Validation accuracy: {results[1]}")

def main():
    args = parse_arguments()
    training_data = prepare_data(args.train_csv, args.im_height, args.im_width, True)
    val_data_dict = {name: prepare_data(path, args.im_height, args.im_width, True) for name, path in parse_val_csvs(args.val_csvs).items()}

    # compute class weights
    if args.balance:
        class_weights = get_weights(args.train_csv)
    else:
        class_weights = None

    # batching and augmentation
    training_data = batch_and_augment(training_data, args.batch_size, augment=args.augment)

    # build pretrained CNN
    model = build_pretrained_cnn(args.im_height, args.im_width)

    # train CNN
    history = fit_pretrained_cnn(model, training_data, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, balance=class_weights)

    # unfreeze top layers and train more
    for layer in model.layers[-20:]:
        layer.trainable = True

    history = fit_pretrained_cnn(model, training_data, epochs=args.finetune_epochs, batch_size=args.batch_size, learning_rate=args.finetune_lr, balance=class_weights)

    # validate CNN
    for name, val_data in val_data_dict.items():
        print(f"Evaluating on {name} validation set:")
        evaluate_cnn(model, val_data)

    # save model
    model.save(args.save_path)


if __name__ == "__main__":
    main()