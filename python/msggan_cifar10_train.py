""" This module contains the code that was used to train the MSG-GAN model
on the CIFAR-10 dataset.  """

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from msggan.datasets import load_images, make_dataset
from msggan.models import MsgProGan

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # stop tensorflow noise


def train_model():

    # Specifications for the training setup
    BATCH_SIZE = 32
    DEPTH = 4
    IMG_HEIGHT = 2**(DEPTH + 1)
    IMG_WIDTH = 2**(DEPTH + 1)
    CHANNELS = 3

    # Download and prepear data
    print("Load dataset")
    cifar10 = tf.keras.datasets.cifar10
    (train_images, _), (test_images, _) = cifar10.load_data()
    train_images = np.r_[train_images, test_images]

    train_dataset = make_dataset(
        train_images, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, DEPTH)
    print("Datasets complete. Initiating model...")

    # Create the model and train it
    model = MsgProGan(
        channels=CHANNELS, folder="python/saved_models/msggan_cifar10_unconditionalEMA", use_eql=True, depth=DEPTH)

    try:
        model.load_model()
        print(f"Model pretrained for {model.epochs[-1]} epochs ")
    except:
        print("Pretrained model is not loaded")
    finally:
        print("Model initiated. Begin training...")
        model.train(train_dataset, 200)


if __name__ == "__main__":
    train_model()
    print("Training complete")
