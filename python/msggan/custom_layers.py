""" All custom layers used in the MSG-GAN. 

This implementation is used to produce the results my thesis, spring 2020.
The implementation builds on the paper and code of MSG-GAN https://arxiv.org/abs/1903.06048
and ProGAN https://arxiv.org/abs/1710.10196. 

"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers


# Normalization layers
class PixelwiseNorm(tf.keras.layers.Layer):
    """ Pixelwise feature vector normalization
    Normalizes each axtivation based on all activations in the same image and location,
    but cross all the channels. (From ProGan) """

    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def call(self, x, alpha=1e-8):
        """ x: inputs
            alpha: smoothing term to avoid devision by zero """
        y = tf.math.sqrt(tf.math.reduce_mean(
            x**2, axis=-1, keepdims=True) + alpha)
        y = x / y
        return y


class MinibatchStdDev(tf.keras.layers.Layer):
    """ Minibatch standard deviation layer for the discriminator """

    def __init__(self):
        """ Derived class constructor """
        super().__init__(name="")

    def call(self, x, alpha=1e-8):
        """ Forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map """

        # [H, W]
        # Compute the std. of every pixel in the batch
        y = tf.math.reduce_std(x, axis=[0, -1])

        # []
        # Average over all the stds in the batch
        y = tf.math.reduce_mean(y) + alpha

        # [B x H x W, 1]
        # Prepear the minibatch standard deviation as a feature map
        y = tf.ones_like(x[:, :, :, :1]) * y

        # [B x H x W x C]
        # Append the new feature map to the input, so the discriminator
        # Have easy access to information about the variaty of images
        # the generator produces.
        return tf.concat([x, y], axis=-1)


# Equalized learning rate layers
class _customConv2DTranspose(tf.keras.layers.Layer):
    """ Implements the equalized learning rate """

    def __init__(self, filters, kernel_size, strides, padding):
        super(_customConv2DTranspose, self).__init__(name="")

        self.kernel_size = kernel_size
        self.filters = filters
        self.stride = strides
        self.padding = padding

    def build(self, input_shape):
        """ Build the weights first when given data """
        self.w = self.add_weight(shape=(*self.kernel_size, input_shape[-1], self.filters),
                                 initializer=tf.random_normal_initializer(
                                     0, 1),
                                 trainable=True, name="w")

        self.b = self.add_weight(shape=(self.filters,),
                                 initializer=tf.zeros_initializer(),
                                 trainable=True, name="b")

        # Calculate the scale to use in the forward pass
        self.fan_in = np.prod(self.kernel_size) * input_shape[-1]
        self.scale = np.sqrt(2 / self.fan_in)

    def call(self, x):
        """ Forward pass of the equalized learning rate layer.
        x: input """

        # Use tf.shape to deal with dynamic batch sizes
        batch_size = tf.shape(x)[0]
        output_shape = [batch_size, *self.kernel_size, self.filters]

        # Scale the weights at runtime
        weights = self.w * self.scale
        return tf.nn.conv2d_transpose(x, weights, output_shape, self.stride, self.padding) + self.b


class _customConv2D(tf.keras.layers.Layer):
    """ Implements the equalized learning rate """

    def __init__(self, filters, kernel_size, strides, padding):
        super(_customConv2D, self).__init__(name="")

        self.kernel_size = kernel_size
        self.filters = filters
        self.stride = strides
        self.padding = padding

    def build(self, input_shape):
        """ Build the weights first when given data """
        self.w = self.add_weight(shape=(*self.kernel_size, input_shape[-1], self.filters),
                                 initializer=tf.random_normal_initializer(
                                     0, 1),
                                 trainable=True, name="w")

        self.b = self.add_weight(shape=(self.filters,),
                                 initializer=tf.zeros_initializer(),
                                 trainable=True, name="b")

        self.fan_in = np.prod(self.kernel_size) * input_shape[-1]
        self.scale = np.sqrt(2 / self.fan_in)

    def call(self, x):

        # Scale the weights at runtime
        weights = self.w * self.scale
        return tf.nn.conv2d(x, weights, self.stride, self.padding) + self.b


class _customDense(tf.keras.layers.Layer):
    """ Implements the equalized learning rate """

    def __init__(self, filters):
        super(_customDense, self).__init__(name="")

        self.filters = filters

    def build(self, input_shape):
        """ Build the weights first when given data """
        self.w = self.add_weight(shape=(input_shape[-1], self.filters),
                                 initializer=tf.random_normal_initializer(
                                     0, 1),
                                 trainable=True, name="w")

        self.b = self.add_weight(shape=(self.filters,),
                                 initializer=tf.zeros_initializer(),
                                 trainable=True, name="b")

        self.fan_in = input_shape[-1]
        self.scale = np.sqrt(2 / self.fan_in)

    def call(self, x):

        # Scale the weights at runtime
        weights = self.w * self.scale
        return tf.matmul(x, weights) + self.b


# RGB converters
class RgbConverter(tf.keras.Model):
    """ Basic RGB-converter. Can be used as both to-rgb and from-rgb. """

    def __init__(self, channels, use_eql):
        super().__init__()

        self.channels = channels
        self.use_eql = use_eql

        # Convolutional layer
        if self.use_eql:
            self.conv2a = _customConv2D(self.channels, (1, 1),
                                        strides=[1, 1, 1, 1], padding="SAME")
        else:
            self.conv2a = layers.Conv2D(self.channels, (1, 1),
                                        strides=(1, 1), padding="SAME")

    def call(self, x):
        """ Forward pass.
        Convert input to/from RGB mode """

        return self.conv2a(x)


class GenInitialDenseBlock(tf.keras.Model):
    """ The first block of the generator.
    Uses dense layer -> 3x3conv instead of transposed convolutions.
    Like the original progressive gan. """

    def __init__(self, filters, use_eql):
        super(GenInitialDenseBlock, self).__init__()

        if use_eql:
            self.dense = _customDense(filters * 16)
            self.conv2a = _customConv2D(
                filters, (3, 3), strides=[1, 1, 1, 1], padding="SAME")
        else:
            self.dense = layers.Dense(filters)
            self.conv2a = layers.Conv2D(
                filters, (3, 3), strides=[1, 1, 1, 1], padding="SAME")

        self.reshape = layers.Reshape((4, 4, filters))
        self.pixnorm = PixelwiseNorm()
        self.lrelu = layers.LeakyReLU(0.2)

    def call(self, x):
        """ Forward pass of the block. """

        # Transform latent vector to image
        y = self.dense(x)
        y = self.reshape(y)
        y = self.pixnorm(self.lrelu(y))

        y = self.conv2a(y)
        y = self.pixnorm(self.lrelu(y))
        return y

# Generator


class GenInitialBlock(tf.keras.Model):
    """ Is used as the first block of the generator """

    def __init__(self, filters, use_eql):
        super(GenInitialBlock, self).__init__()

        if use_eql:
            self.conv2a = _customConv2DTranspose(
                filters, (4, 4), strides=[1, 1, 1, 1], padding="VALID")
            self.conv2b = _customConv2D(
                filters, (3, 3), strides=[1, 1, 1, 1], padding="SAME")

        else:
            self.conv2a = layers.Conv2DTranspose(
                filters, (4, 4), strides=(1, 1), padding="VALID")
            self.conv2b = layers.Conv2D(
                filters, (3, 3), strides=(1, 1), padding="SAME")

        self.pixnorm = PixelwiseNorm()
        self.lrelu = layers.LeakyReLU(0.2)

    def call(self, x):
        """ Forward pass of the block.
        :param x: input
        :return: y => output
        """

        y = self.lrelu(self.conv2a(x))

        # Pixelwise normalization after 3x3 conv
        y = self.pixnorm(self.lrelu(self.conv2b(y)))
        return y


class GenGeneralConvBlock(tf.keras.Model):
    """ Used as a general convolutional block for the generator """

    def __init__(self, filters, use_eql):
        super(GenGeneralConvBlock, self).__init__()

        if use_eql:
            self.conv2a = _customConv2D(
                filters, (3, 3), strides=[1, 1, 1, 1], padding="SAME")
            self.conv2b = _customConv2D(
                filters, (3, 3), strides=[1, 1, 1, 1], padding="SAME")

        else:
            self.conv2a = layers.Conv2D(
                filters, (3, 3), strides=(1, 1), padding="SAME")
            self.conv2b = layers.Conv2D(
                filters, (3, 3), strides=(1, 1), padding="SAME")

        self.upsample = layers.UpSampling2D(size=(2, 2))
        self.pixnorm = PixelwiseNorm()
        self.lrelu = layers.LeakyReLU(0.2)

    def call(self, x):
        """ Forward pass of the block.
        :param x: input
        :return: y => output
        """

        y = self.upsample(x)

        y = self.pixnorm(self.lrelu(self.conv2a(y)))
        y = self.pixnorm(self.lrelu(self.conv2b(y)))
        return y


class DisFinalBlock(tf.keras.Model):
    """ Final block for the Discriminator """

    def __init__(self, out_channels, use_eql):
        """
        constructor of the class
        :param out_channels: number of input channels """
        super().__init__()

        # declare the required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()

        if use_eql:
            self.conv2a = _customConv2D(
                out_channels, (3, 3), strides=[1, 1, 1, 1], padding="SAME")
            self.conv2b = _customConv2D(
                out_channels, (4, 4), strides=[1, 1, 1, 1], padding="VALID")

            # final conv layer emulates a fully connected layer
            self.conv2c = _customConv2D(
                1, (1, 1), strides=[1, 1, 1, 1], padding="SAME")

        else:
            self.conv2a = layers.Conv2D(
                out_channels, (3, 3), strides=(1, 1), padding="SAME")
            self.conv2b = layers.Conv2D(
                out_channels, (4, 4), strides=(1, 1), padding="VALID")

            # final conv layer emulates a fully connected layer
            self.conv2c = layers.Conv2D(
                1, (1, 1), strides=(1, 1), padding="SAME")

        # leaky_relu:
        self.lrelu = layers.LeakyReLU(0.2)

    def call(self, x):
        """
        forward pass of the FinalBlock
        :param x: input
        :return: y => output
        """
        # Minibatch standard deviation layer
        y = self.batch_discriminator(x)

        y = self.lrelu(self.conv2a(y))
        y = self.lrelu(self.conv2b(y))

        # Fully connected layer
        # This layer has linear activation (wasserstein loss)
        y = self.conv2c(y)

        # Flatten the output raw discriminator scores
        return tf.reshape(y, [-1])
