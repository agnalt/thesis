""" This module contains the code to create an MSG-GAN and train it. 

This implementation is used to produce the results my thesis, spring 2020.
The implementation builds on the paper and code of MSG-GAN https://arxiv.org/abs/1903.06048
and ProGAN https://arxiv.org/abs/1710.10196. 

"""


import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tqdm import tqdm

from msggan.custom_layers import (DisFinalBlock, DisGeneralConvBlock,
                                  GenGeneralConvBlock, GenInitialBlock,
                                  GenInitialDenseBlock, RgbConverter)
from msggan.utils import adjust_color_range, create_path, pbar

# import tensorflow_addons as tfa


class MsgProGan(tf.keras.Model):

    def __init__(self, channels=3, folder="saved_models/msggan_test", use_eql=True, depth=4, use_dense=False):
        super().__init__()

        self.use_eql = use_eql
        self.use_dense = use_dense
        self.depth = depth

        self.generator = self.make_generator(channels)
        self.discriminator = self.make_discriminator(channels)
        self.generator_ema = self.make_generator(channels)  # Shadow model

        self.noise_dim = 512
        self.seed = self.normalized_noise(10)
        self.mse_seed = self.normalized_noise(32)
        self.use_gp = True
        self.batch_size = None
        self.drift = 0.001

        # Folder to store model data
        self.folder = folder
        create_path(self.folder)

        # Model metrics
        self.g_train_loss = []
        self.d_train_loss = []
        self.mse_training = []
        self.epochs = [0]

        # Optimizers
        # self.generator_optimizer = tf.keras.optimizers.RMSprop(0.003)
        # self.discriminator_optimizer = tf.keras.optimizers.RMSprop(0.003)

        self.generator_optimizer = tf.keras.optimizers.Adam(
            0.003, beta_1=0, beta_2=0.99, epsilon=1e-8)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            0.003, beta_1=0, beta_2=0.99, epsilon=1e-8)



    def save_model(self):
        """ Save the state of the model. """

        # Save directory
        save_dir = self.folder + "/saved_model"
        if not os.path.exists(save_dir):
            os.makedirs(f"{save_dir}")

        # Save weights
        epoch = self.epochs[-1]
        self.generator.save_weights(
            save_dir + f"/checkpoints/generator/gen-{epoch:04d}")
        self.generator_ema.save_weights(
            save_dir + f"/checkpoints/generator/gen_ema-{epoch:04d}")
        self.discriminator.save_weights(
            save_dir + f"/checkpoints/discriminator/disc-{epoch:04d}")

        # Training stats
        g_train_loss = np.array(self.g_train_loss)
        d_train_loss = np.array(self.d_train_loss)
        mse_training = np.array(self.mse_training)
        epochs = np.array(self.epochs).astype(float)
        seed = self.seed.numpy().reshape(-1, 512)
        mse_seed = self.mse_seed.numpy().reshape(-1, 512)

        data = {"/g_train_loss": g_train_loss,
                "/d_train_loss": d_train_loss,
                "/mse_training": mse_training,
                "/epochs": epochs,
                "/seed": seed,
                "/mse_seed": mse_seed}

        for key, value in data.items():
            np.savetxt(save_dir + key + ".csv", value,
                       fmt="%.6f", delimiter=",")


    def load_model(self, epoch=None):
        """ Load and restore the model to the final training state. """
        save_dir = self.folder + "/saved_model"

        try:
            cutoff = epoch + 1
        except:
            cutoff = None

        # Load stats
        self.g_train_loss = list(np.genfromtxt(
            save_dir + "/g_train_loss.csv", dtype="float32"))[:cutoff]
        self.d_train_loss = list(np.genfromtxt(
            save_dir + "/d_train_loss.csv", dtype="float32"))[:cutoff]
        self.mse_training = list(np.genfromtxt(
            save_dir + "/mse_training.csv", delimiter=",", dtype="float32").reshape(-1, self.depth))[:cutoff]
        self.epochs = list(np.genfromtxt(
            save_dir + "/epochs.csv").astype(int))[:cutoff]
        self.seed = tf.constant(np.genfromtxt(
            save_dir + "/seed.csv", delimiter=",", dtype="float32").reshape(-1, 1, 1, 512))
        self.mse_seed = tf.constant(np.genfromtxt(
            save_dir + "/mse_seed.csv", delimiter=",", dtype="float32").reshape(-1, 1, 1, 512))

        # Load weights
        epoch = self.epochs[-1]
        try:
            self.generator.load_weights(
                save_dir + f"/checkpoints/generator/gen-{epoch:04d}")
            self.discriminator.load_weights(
                save_dir + f"/checkpoints/discriminator/disc-{epoch:04d}")
            try:
                self.generator_ema.load_weights(
                    save_dir + f"/checkpoints/generator/gen_ema-{epoch:04d}")
            except:
                print("No ema generator model.")
        except:
            print("It is only possible to load every 20th epoch.")
            # print("Loading last saved model")
            # self.load_model()
            # print(f"Model trained for {epoch} epochs is loaded.")

        self.mse_imgs = self.generator_ema(self.mse_seed, training=False)

    def train(self, dataset, epochs):
        """ The training loop of the MSG-GAN """

        # Corny way of calculating the total number of images in dataset
        dataset_list = list(dataset)
        num_batches = len(dataset_list)
        batch_size = dataset_list[0][0].shape[0]
        num_in_last_batch = dataset_list[-1][0].shape[0]
        total_imgs = (num_batches - 1) * batch_size + num_in_last_batch

        # Calculate the MSE to check for convergence
        if self.epochs[-1] == 0:
            self.mse_imgs = self.generator_ema(self.mse_seed, training=False)

        # Progress stats
        start = self.epochs[-1]
        end = start + epochs
        bar = pbar(total_imgs * epochs, start, end)

        # Training loop
        for epoch in range(start, end):

            # Keep track of training progress and stats
            start = time.time()
            gen_epoch_loss = []
            disc_epoch_loss = []

            # Go through batches
            for image_batch in dataset:
                self.batch_size = image_batch[0].shape[0]

                # Train on batch.
                gen_loss, disc_loss = self.train_step(image_batch)

                # Record loss.
                gen_epoch_loss.append(gen_loss.numpy())
                disc_epoch_loss.append(disc_loss.numpy())

                # Update exponential moving average
                self.update_ema(self.generator.get_weights())

                # Update progressbar
                bar.update(self.batch_size)

            # Update loss metrics
            self.g_train_loss.append(np.mean(gen_epoch_loss))
            self.d_train_loss.append(np.mean(disc_epoch_loss))
            self.calc_mse()

            # Produce images for the GIF as we go
            self.generate_and_save_images(self.seed)

            tf.print('\nTime for epoch {} is {} min'.format(
                epoch + 1, (time.time()-start) / 60))

            self.epochs.append(epoch + 1)

            # Save the model every 20th epoch
            if self.epochs[-1] % 400 == 0:
                self.save_model()

        # Generate and save after the final epoch
        self.save_model()
        self.generate_and_save_images(self.seed)

        # Close the progress bar for the finished epoch
        bar.close()
        del bar

        tf.print('\nTime for epoch {} is {} min'.format(
            epoch + 1, (time.time()-start) / 60))

    def update_ema(self, new_weights, decay=0.999):
        """ Keep the shadow model (generator_ema) updated with the new
        parameters. The ema variables are updates the same way
        that is done in
        https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage """

        # Get current ema weights
        ema_weights = self.generator_ema.get_weights()

        # Update every weight on shadow ema generator
        for ema_w, new_w in zip(ema_weights, new_weights):
            ema_w -= (1 - decay) * (ema_w - new_w)

        self.generator_ema.set_weights(ema_weights)

    def calc_mse(self):
        """ Calculate the mean squared error between images in sequential epochs """

        old_imgs = self.mse_imgs
        new_imgs = self.generator_ema(self.mse_seed, training=False)

        # Calculate MSE between images
        mse_sizes = []
        for old_img, new_img in zip(old_imgs, new_imgs):
            mse = tf.reduce_mean((old_img - new_img)**2)
            mse_sizes.append(mse.numpy())

        # Append values to attribute
        self.mse_training.append(mse_sizes)

        # Update images
        self.mse_imgs = new_imgs

    def generate_images(self, num):
        
        batch_size = 32

        if num <= batch_size:
            self.generate_image_batch(batch_no=0, batch_size=num)
        else:
            for i in tqdm(range(num // batch_size), unit="batches"):
                self.generate_image_batch(batch_no=i, batch_size=batch_size)

            if num % batch_size != 0:
                self.generate_image_batch(
                    (num // batch_size)+1, num % batch_size)

    def generate_image_batch(self, batch_no, batch_size=1):
        noise = self.normalized_noise(batch_size)
        images = self.generator_ema(noise, training=False)[-1]

        epoch = self.epochs[-1]
        save_path = self.folder + f"/synthetic_images/epoch_{epoch:03d}"
        create_path(save_path)

        for i, image in enumerate(images):
            # Correct for grayscale images
            if image.shape[-1] == 1:
                image = image[:, :, 0]
                plt.gray()

            image = adjust_color_range(image).numpy()
            plt.imsave(
                save_path + f"/gen_image{batch_no:04d}_{i:02d}.png", image)

    def generate_and_save_images(self, test_input, save=True, f_name=""):
        # Notice `training` is set to False.

        predictions = self.generator_ema(test_input, training=False)

        height = predictions[0].shape[0]
        width = len(predictions)
        _, axs = plt.subplots(height, width, figsize=(width, height))
        if len(axs.shape) == 1:
            axs = np.expand_dims(axs, axis=-1)
        for i in range(height):
            for j in range(width):
                # Make the image suitable for display
                image = predictions[j][i]
                image = adjust_color_range(image)

                # Correct for grayscale
                if image.shape[-1] == 1:
                    image = image[:, :, 0]
                    plt.gray()
                axs[i][j].imshow(image)
                axs[i][j].axis("off")

        if save:
            self.save_im(self.epochs[-1], f_name)

        plt.close()

    def save_im(self, epoch, f_name):
        path_name = f"{self.folder}/training_imgs"
        if not os.path.exists(path_name):
            os.makedirs(f"{path_name}")

        file_name = "{}/image_at_epoch_{:04d}".format(path_name, epoch)

        if f_name != "":
            file_name += f"_{f_name}"
        plt.savefig(file_name + ".png")

    @tf.function
    def train_step(self, images):
        """ Train the discriminator and the generator.  """

        # Noise for the generator
        noise = self.normalized_noise(self.batch_size)

        # Train the discriminator
        with tf.GradientTape() as disc_tape:
            gen_images = self.generator(noise, training=False)

            # Detach the fake images from the computational graph
            gen_images = tuple([tf.stop_gradient(img) for img in gen_images])

            # Forward pass
            disc_loss = self.discriminator_loss(images, gen_images)

            # Backward pass
            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, self.discriminator.trainable_variables)

            # Update weights by applying gradients
            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as gen_tape:
            # Forward pass
            gen_images = self.generator(noise, training=True)
            gen_loss = self.generator_loss(gen_images)

            # Backward pass
            gradients_of_generator = gen_tape.gradient(
                gen_loss, self.generator.trainable_variables)

            # Update weights by applying gradients
            self.generator_optimizer.apply_gradients(
                zip(gradients_of_generator, self.generator.trainable_variables))

        return gen_loss, disc_loss

    @tf.function
    def gradient_penalty(self, images, generated_images, penalty_coeff=10):
        """ Calculate the gradient penalty using GradientTape() """

        batch_size = images[0].shape[0]
        epsilon = tf.random.uniform(
            [batch_size, 1, 1, 1], 0., 1.)  # Random number

        slopes = []
        mixed_imgs = []

        # Create mixed images of real and fake samples
        for imgs, gen_imgs in zip(images, generated_images):

            # Create mixed images
            mixed_img = epsilon * imgs + (1 - epsilon) * gen_imgs
            mixed_imgs.append(mixed_img)
        mixed_imgs = tuple(mixed_imgs)

        # Compute gradients of discriminator w.r.t. the mixed images
        with tf.GradientTape() as t:
            t.watch(mixed_imgs)
            mixed_out = self.discriminator(mixed_imgs, training=True)
            gradients = t.gradient(mixed_out, mixed_imgs)

        # Calculate the 2-norm of the gradient for each image size
        for gradient in gradients:

            # Flatten to a vector
            gradient = tf.reshape(gradient, (gradient.shape[0], -1))

            # Calculate the 2-norm, but add a small number for numeric stability
            slope = tf.math.sqrt(tf.math.reduce_sum(
                gradient ** 2, axis=1) + 1e-10)
            slopes.append(slope)

        # Create a matrix of gradient norms for all image sizes
        slopes = [tf.expand_dims(slope, -1)
                  for slope in slopes]  # [batch_size, 1]
        slopes = tf.concat(slopes, axis=1)  # [batch_size, depth]

        gamma = 1  # The target gradient for wgan
        g_penalty = (slopes - gamma)**2 / gamma**2

        # Calculate the expectation of penalty
        penalty = tf.math.reduce_mean(g_penalty, axis=0)  # For each image size
        penalty = tf.math.reduce_mean(penalty)  # Over the batch

        return penalty * penalty_coeff

    def discriminator_loss(self, images, generated_images):
        """ Implements the Wasserstein loss function for the discriminator """

        # Wasserstein loss
        fake_out = self.discriminator(generated_images, training=True)
        real_out = self.discriminator(images, training=True)

        loss = tf.math.reduce_mean(
            fake_out) - tf.math.reduce_mean(real_out)  # Wasserstein loss
        loss += (self.drift * tf.math.reduce_mean(real_out ** 2))  # Drift loss

        if self.use_gp:
            # Calculate the WGAN-GP (gradient penalty)
            gp = self.gradient_penalty(images, generated_images)
            loss += gp

        return loss

    def generator_loss(self, generated_images, y=None):
        """ Implements the Wasserstein loss for the generator. """
        # Calculate the WGAN loss for generator
        fake_out = self.discriminator(generated_images, training=False)

        loss = - tf.math.reduce_mean(fake_out)
        return loss

    def hypersphere_normalization(self, noise):
        """ Gaussian noise --> Hypersphere surface """
        # Normalize points to lie on unit a hypersphere
        noise = (noise / tf.norm(noise, axis=-1, keepdims=True))

        # Scale by a factor of sqrt(dims)
        return noise * (noise.shape[-1]**0.5)

    def random_noise(self, batch_size=10):
        """ Generate a batch of Gaussian noise """
        # Draw from a normal distribution
        noise = tf.random.normal(
            [batch_size, 1, 1, self.noise_dim], mean=0, stddev=1)
        return noise

    def normalized_noise(self, batch_size=1):
        """ Generate normalized noise """

        # Draw from normal distribution
        noise = self.random_noise(batch_size)

        # Normalize point to hypersphere, and scale by sqrt(dim)
        noise = self.hypersphere_normalization(noise)
        return noise

    def interpolate(self, transitions=10):
        """ Interpolate between to random points in latent space. """

        # Generate two latent vectors to interpolate between
        noise = self.random_noise(2)

        # Interpolate and normalize
        interpolates = tf.cast(np.linspace(
            noise[0], noise[1], 10, axis=0), dtype=tf.float32)
        interpolates = self.hypersphere_normalization(interpolates)

        images = self.generator_ema(interpolates, training=False)[-1]
        return images

    def make_generator(self, channels=3):
        """ Create the generator architecture """

        inputs = layers.Input(shape=(1, 1, 512))

        # Maybe use dense layer(?) -> (3x3)conv instead of convT. (experimental)
        if self.use_dense:
            blocks = [GenInitialDenseBlock(filters=512, use_eql=self.use_eql)]
        else:
            blocks = [GenInitialBlock(
                filters=512, use_eql=self.use_eql)]  # Default

        rgb_converters = [RgbConverter(channels, use_eql=self.use_eql)]

        # Define the number of filters in every layer of the model
        filters = [512, 512, 512, 256, 128, 64, 32, 16]
        for n_filters in filters[:self.depth-1]:

            block = GenGeneralConvBlock(
                filters=n_filters, use_eql=self.use_eql)  # Upsampling
            rgb = RgbConverter(channels, use_eql=self.use_eql)

            blocks.append(block)
            rgb_converters.append(rgb)

        x = inputs
        outputs = []

        # MSG generator architecture. Output images at every scale.
        for block, rgb_converters in zip(blocks, rgb_converters):
            x = block(x)
            img = rgb_converters(x)

            outputs.append(img)

        outputs = tuple(outputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)


    # Alternative discriminator that use 
    # only one RGB-converter in the first block

    # def make_discriminator(self, channels=3):

    #     dims = [(2**(i+2),)*2 for i in range(9)]

    #     # out_channels = [512, 512, 512, (512, 256), (256, 128),
    #     #                 (128, 64), (64, 32), (32, 16)]  # All exept the final block (512)
    #     out_channels = [512, 512, 512, (256, 512), (128, 256),
    #                     (64, 128), (32, 64), (16, 32)]  # All exept the final block (512)

    #     # Create lists of the layers
    #     inputs = [tf.keras.Input(shape=(*dim, channels)) for dim in dims[:self.depth]]
    #     blocks = [DisFinalBlock(out_channels=512, use_eql=self.use_eql)]  # Final block

    #     for out_ch in out_channels[:self.depth - 1]:
    #         block = DisGeneralConvBlock(out_channels=out_ch, use_eql=self.use_eql)
    #         blocks.append(block)

    #     x = None
    #     inputs_reversed = reversed(inputs)
    #     blocks = reversed(blocks)

    #     # MSG discriminator architecture. Take in images at every scale.
    #     for image_in, block in zip(inputs_reversed, blocks):

    #         # Send image through the highest res. block of the discriminator
    #         if x == None:
    #             # For the first block, use from_rgb converter
    #             try:
    #                 from_rgb_c = out_channels[self.depth - 2][0]
    #             except:
    #                 from_rgb_c = out_channels[self.depth - 2]

    #             x = RgbConverter(from_rgb_c, use_eql=self.use_eql)(image_in)
    #             x = block(x)

    #         # Concatenate every new image with output of previous blocks
    #         else:
    #             x = tf.concat([image_in, x], axis=3)
    #             x = block(x)

    #     # Output a single prediction for each sample
    #     return tf.keras.Model(inputs=inputs, outputs=x)

    # Model B
    def make_discriminator(self, channels=3):
        """ Create the discriminator architecture """

        dims = [(2**(i+2),)*2 for i in range(9)]

        # (in_channels, out_channels)
        out_channels = [(512, 256), (512, 256), (512, 256), (256, 256), (128, 128),
                        (64, 64), (32, 32), (16, 16)]  # All exept the final block (512)

        # Create lists of the layers
        inputs = [tf.keras.Input(shape=(*dim, channels))
                  for dim in dims[:self.depth]]
        # Final block
        blocks = [DisFinalBlock(out_channels=512, use_eql=self.use_eql)]
        rgb_converters = [RgbConverter(
            out_channels[0][1], use_eql=self.use_eql)]

        for out_ch in out_channels[:self.depth - 1]:
            block = DisGeneralConvBlock(
                out_channels=out_ch, use_eql=self.use_eql)
            rgb_converter = RgbConverter(out_ch[0] // 2, use_eql=self.use_eql)

            blocks.append(block)
            rgb_converters.append(rgb_converter)

        x = None
        inputs_reversed = reversed(inputs)
        blocks = reversed(blocks)
        rgb_converters = reversed(rgb_converters)

        # MSG discriminator architecture. Take in images at every scale.
        for image_in, block, converter in zip(inputs_reversed, blocks, rgb_converters):

            rgb_features = converter(image_in)
            try:
                # This will fail for the first input
                x = tf.concat([rgb_features, x], axis=3)
            except:
                # No previous activations to concat with. Use only rgbfeatures.
                x = rgb_features
            finally:
                x = block(x)

        # Output a single prediction for each sample
        return tf.keras.Model(inputs=inputs, outputs=x)

