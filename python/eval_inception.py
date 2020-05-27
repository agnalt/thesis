""" This module is used for evaluating the inception model for retrieve the logits and 
features needed to compute the inception score and frechet inception distance.

 """

import argparse
import os
import sys

import numpy as np
import scipy.stats
import tensorflow as tf
import tensorflow_gan as tfgan
from tqdm import tqdm

parser = argparse.ArgumentParser(description='The split')
parser.add_argument("--index")

args = parser.parse_args()
i = args.index


os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # Stop tensorflow warnings




def save_array(save_dir, fname, array):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(f"{save_dir}/{fname}.csv", "a") as f:
        np.savetxt(f, array)


def inception_output(batch):
    """ Takes in batch of images [batch_size, hight, width, 3] 
    and return the conditional probability and the activations
    from the pool_3 layer of the inception model. """

    batch = tf.constant(batch)

    output = tfgan.eval.run_inception(batch)
    return output["logits"].numpy(), output["pool_3"].numpy()


def inception_stats(path, images, index, name="test"):
    """ Get stats from the inception model. """

    # limit = 5000  # Maximum number of image feature in each file

    save_dir = path
    save_logits = f"{name}_inception_logits_{index:03}"
    save_features = f"{name}_inception_features_{index:03}"

    batch_size = 100
    batched_imgs = tf.data.Dataset.from_tensor_slices(
        images).shuffle(5000).batch(batch_size, drop_remainder=False)

    print("\nForward pass of the inception model")

    pbar = tqdm(total=len(images), unit="images")
    for i, batch in enumerate(batched_imgs):
        logit, feature = inception_output(batch)

        # Write the arrays to files
        save_array(save_dir + "/logits", save_logits, logit)
        save_array(save_dir + "/features", save_features, feature)

        pbar.update(batch_size)


def load_images(path, img_height, img_width, split=(0, 5000)):
    # Create a list of the image paths
    img_paths = []
    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        img_paths.append(img_path)

    img_paths = img_paths[split[0]:split[1]]

    print("Loading images")
    train_images = []
    for img_path in tqdm(img_paths, unit="images", file=sys.stdout):

        # Read, decode, convert and resize the image
        try:
            img = tf.io.read_file(img_path)
            img = tf.io.decode_image(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, [img_height, img_width])
            train_images.append(img)
        except:
            print("load_image: Error, file does not satisfy requirements")
    train_images = tf.stack(train_images, axis=0)

    return train_images

# Change the directory so it matches the model and directory with generated images.
# path = "python/saved_models/msggan_cifar10_unconditional4"

# images = load_images(path + "/synthetic_images/epoch_200", 32, 32)


N = 5000
# total_splits = images.shape[0] // N

i = int(i)
print(f"\nSplit {i}")
# split = images[N * i:N * (i+1)]
split = (N * i, N * (i + 1))


images = load_images(
    "python/saved_models/msggan_nemo_sediment/synthetic_images/epoch_1200", 128, 128, split)
inception_stats("python/saved_models/msggan_nemo_sediment/eval/epoch_1200",
                images, i, "sediment_1200")

# images = load_images("python/data/nemo/benthic_train", 128, 128, split)
# inception_stats("python/data/nemo", images, i, "benthic_train")

# images = load_images("python/data/nemo/planktic_train", 128, 128, split)
# inception_stats("python/data/nemo", images, i, "planktic_train")

# images = load_images("python/data/nemo/agglutinated_train", 128, 128, split)
# inception_stats("python/data/nemo", images, i, "agglutinated_train")

# images = load_images("python/data/nemo/sediment_train", 128, 128, split)
# inception_stats("python/data/nemo", images, i, "sediment_train")

# from msggan.datasets import load_images

# cifar10 = tf.keras.datasets.cifar10
# (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# train_images = np.r_[train_images, test_images]
# images = train_images
