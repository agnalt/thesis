""" This module contains the code to train a basic GAN to learn the
distribution of the unit circle.

Use GPU for performance boost.

 """


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tqdm import tqdm

tf.random.set_seed(0)
np.random.seed(0)



def make_discriminator(n_inputs=2):
    """ GAN discriminator
    n_inputs: dimensionality of input data """

    model = tf.keras.Sequential()
    model.add(layers.Dense(256, activation='relu',
                           kernel_initializer='he_uniform',
                           input_dim=n_inputs))

    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile the discriminator
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model


def make_generator(latent_dim, n_outputs=2):
    """ GAN generator
    latent_dim: dimensionality of input variable z
    n_outputs: dimensionality of output """

    model = tf.keras.Sequential()
    model.add(layers.Dense(256, activation="relu",
                           kernel_initializer="he_uniform",
                           input_dim=latent_dim))
    model.add(layers.Dense(n_outputs, activation="linear"))

    return model


def make_gan(generator, discriminator):
    """ Combined generator and discriminator. """

    # Disable training of discriminator as default
    discriminator.trainable = False

    # Create GAN
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)

    # Compile the GAN
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


def real_distribution(n=200, r=1):
    """ Distribution of the real training data.
    A circle with radius r. """

    theta = np.linspace(0, 2*np.pi, n)

    x1 = r * np.sin(theta)
    x2 = r * np.cos(theta)

    return x1, x2


def generate_real_samples(n):
    """ Choose n random training points. """

    # Generate the population
    population = 1000000
    x1, x2 = real_distribution(population)
    X = np.c_[x1, x2]

    # Pick a subsample
    i = np.random.randint(0, population, n)
    X = X[i]

    # Generate class labels
    y = np.ones((n, 1))

    return X, y


def generate_latent_points(latent_dim, n):
    """ Sample points from latent space of the generator. """

    # Sample from uniform distribution
    x_input = np.random.uniform(-1, 1, latent_dim * n)

    # Make batch
    x_input = x_input.reshape(n, latent_dim)

    return x_input


def generate_fake_samples(generator, latent_dim, n):
    """ Generate fake samples with class labels """

    # Sample points from latent space
    x_input = generate_latent_points(latent_dim, n)

    # Forward pass
    X = generator.predict(x_input)

    # Generate labels for the discriminator
    y = np.zeros((n, 1))

    return X, y


def train(g_model, d_model, gan_model, latent_dim, training_data):
    """ The completion of one whole training epoch."""

    for batch in data:

        batch_size = batch.shape[0]

        # Real samples
        x_real = batch

        # Fake samples
        z = generate_latent_points(latent_dim, batch_size)
        x_fake = generator.predict(z)

        # Train discriminator
        # Send in real and fake samples with 1 and 0 as labels
        d_model.train_on_batch(x_real, np.ones((batch_size, 1)))
        d_model.train_on_batch(x_fake, np.zeros((batch_size, 1)))

        # Train the generator
        # Update the generator via the discriminator's error 
        # with flipped labels
        gan_model.train_on_batch(z, np.ones((batch_size, 1)))

if __name__ =="__main__":

    # Make training dataset
    X, y = generate_real_samples(300)

    data = tf.data.Dataset.from_tensor_slices(X)
    data = data.batch(32)

    # Number of input dims
    latent_dim = 1

    # Create models
    discriminator = make_discriminator()
    generator = make_generator(latent_dim)
    gan_model = make_gan(generator, discriminator)

    # Training stats
    train_plot = []
    train_loss = []
    i = 0

    # Train the GAN
    epochs = 1200

    # Training loop
    for i in tqdm(range(i, i + epochs)):
        train(generator, discriminator, gan_model, latent_dim, data)

    # ... additional code to generate figures and evaluate the gan
    # can be found along with additional source code on github...
