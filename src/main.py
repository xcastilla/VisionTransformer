import argparse
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import yaml
from typing import Dict, Any, Tuple

from model import *

def load_data() -> Tuple[Any, Any, int]:
    """ Load CIFAR-100 train and test splits """
    (x_train, y_train),  (x_test, y_test)= tf.keras.datasets.cifar100.load_data()
    num_classes = 100
    return (x_train, y_train), (x_test, y_test), num_classes

def train(config: Dict[str, Any]) -> None:
    # Load CIFAR-100
    train_data, test_data, num_classes = load_data()
    x_train, y_train = train_data
    x_test, y_test = test_data

    # Normalize train and test data
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    x_mean = np.mean(x_train, axis = 0)
    x_std = np.std(x_train, axis = 0)
    x_train -= x_mean
    x_train /= x_std
    x_test -= x_mean
    x_test /= x_std

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # Create ViT Model
    vit = ViT(config, output_size=num_classes)
    x = tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    model = tf.keras.Model(inputs=[x], outputs=vit.call(x))
    vit.data_augmentation.layers[0].adapt(x_train)

    # Create optimizer and prepare training metrics and callbacks
    optimizer = tfa.optimizers.AdamW(
        learning_rate=config['training_options']['learning_rate'],
        weight_decay= config['training_options']['weight_decay']
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    model.summary()
    checkpoint_filepath = config['training_options']['checkpoint_dir']
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    # Model training
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=config['training_options']['batch_size'],
        epochs=config['training_options']['train_epochs'],
        validation_split=config['training_options']['validation_split'],
        callbacks=[checkpoint_callback],
    )

    # Load best model and display final accuracy of the model in the test set
    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)

    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")


def parse_conf(config_file: str) -> Dict[str, Any]:
    """ Parse YAML config file to a dictionary """
    with open(config_file) as f:
        options = yaml.safe_load(f)
    return options

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Path to YAML configuration file with training and model options.', required=True)

    args = parser.parse_args()
    options = parse_conf(args.config)
    train(options)
        
    