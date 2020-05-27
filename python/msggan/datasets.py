""" This module is used to prepear the datasets needed for MSG-GAN training. """

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tqdm import tqdm


def load_images(path, img_height, img_width):
    """ Load a directory of images.
    return: a large tensor with loaded images. """

    # Create a list of the image paths
    img_paths = []
    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        img_paths.append(img_path)

    print("Loading images")
    train_images = []
    for img_path in tqdm(img_paths, unit="images", file=sys.stdout):

        # Read, decode, convert and resize the image
        img = tf.io.read_file(img_path)
        img = tf.io.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [img_height, img_width])
        train_images.append(img)

    train_images = tf.stack(train_images, axis=0)

    return train_images


def make_dataset(images, batch_size, img_height, img_width, depth):
    """ Take a set of images and create a dataset for GAN training.

    images: list of images [img(h, w, c), img...]
    returns: the dataset """

    # Expand to one channel if the images are grayscale
    if len(images[0].shape) == 2:  # Only (H x W)
        images = tf.expand_dims(images, axis=-1)

    channels = images[0].shape[-1]
    if not (channels == 1 or channels == 3):
        print("Not grayscale or RGB data. Got {}, expected 1 or 3".format(channels))
        return None

    images = tf.image.convert_image_dtype(images, tf.float32)

    images = tf.image.resize(images, [img_height, img_width])  # Resize
    images = (images - 0.5) / 0.5  # Normalize the images to [-1, 1]

    BUFFER_SIZE = images.shape[0]

    # Batch and shuffle the data
    dataset = tf.data.Dataset.from_tensor_slices(
        images).shuffle(BUFFER_SIZE).batch(batch_size)

    def image_reshape(depth):
        """ Creates batches for the msgGAN: """
        def img_reshape(x):

            sizes = [2**(i + 1) for i in range(1, depth+1)]
            shapes = [(dim, dim) for dim in sizes]

            image_scales = []
            for shape in shapes:
                image_scales.append(tf.image.resize(x, shape))

            return image_scales
        return img_reshape

    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.map(image_reshape(depth))
    dataset = dataset.cache()
    dataset = dataset.shuffle(len(images))
    dataset = dataset.batch(batch_size, drop_remainder=False)

    return dataset
