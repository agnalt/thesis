
""" This module contains the code that was used to train the MSG-GAN on the
unconditional dataset of foraminifera. """

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from msggan.datasets import load_images, make_dataset
from msggan.models import MsgProGan

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"



def train_model():

    # Specifications for the training setup
    BATCH_SIZE = 8
    DEPTH = 6
    IMG_HEIGHT = 2**(DEPTH + 1)
    IMG_WIDTH = 2**(DEPTH + 1)
    CHANNELS = 3

    train_images = load_images("python/data/nemo/all", IMG_HEIGHT, IMG_WIDTH)
    train_dataset = make_dataset(
        train_images, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, DEPTH)
    print("Datasets complete. Initiating model...")

    print(
        f"Model depth: {DEPTH}, channels: {CHANNELS}, image resolution: {IMG_HEIGHT}, batch size: {BATCH_SIZE}")
    # Create the model and train it
    model = MsgProGan(
        channels=CHANNELS, folder="python/saved_models/msggan_nemo_unconditionalEMA2", use_eql=True, depth=DEPTH)

    try:
        model.load_model()
        print(f"Model pretrained for {model.epochs[-1]} epochs ")
    except:
        print("Pretrained model is not loaded")
    finally:
        print("Model initiated. Begin training...")
        model.train(train_dataset, 1000)


if __name__ == "__main__":
    train_model()
    print("Training complete")

