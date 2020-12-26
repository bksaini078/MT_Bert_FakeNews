import math
import numpy as np 
import tensorflow as tf
from  tensorflow.keras.layers  import *

# Edited files with weight normalization and mean only batch normalization
from PI_model.weight_norm_layers.Conv2D import Conv2D
from PI_model.weight_norm_layers.Dense import Dense

def create_embedding(x):
    # x = tf.make_ndarray(x)
    x=tf.expand_dims(x, -1)
    # x=tf.reshape(x, (x.shape)+ [1], name=None)
    # tf.convert_to_tensor(x )
    return x

class PiModel(tf.keras.Model):
    """ Class for defining eager compatible tfrecords file

        I did not use tfe.Network since it will be depracated in the
        future by tensorflow.
    """

    def __init__(self,args, vocab_size):
        """ Init

            Set all the layers that need to be tracked in the process of
            gradients descent (pooling and dropout for example dont need
            to be stored)
        """
        self.max_len = args.max_len
        self.vocab_size= vocab_size

        super(PiModel, self).__init__()
        self._emb= Embedding(vocab_size, 128, input_length=None)
        
        self._conv1a = Conv2D(filters=128,kernel_size=[3,2],padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1),kernel_initializer=tf.keras.initializers.he_uniform(),
                                                        bias_initializer=tf.keras.initializers.constant(
                                                            0.1),
                                                        weight_norm=True, mean_only_batch_norm=True)
        self._conv1b =Conv2D(filters=128, kernel_size=[3,3],
                                                        padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                                        kernel_initializer=tf.keras.initializers.he_uniform(),
                                                        bias_initializer=tf.keras.initializers.constant(
                                                            0.1),
                                                        weight_norm=True, mean_only_batch_norm=True)
        self._conv1c = Conv2D(filters=128,kernel_size=[3,3],
                                                        padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                                        kernel_initializer=tf.keras.initializers.he_uniform(),
                                                        bias_initializer=tf.keras.initializers.constant(
                                                            0.1),
                                                        weight_norm=True, mean_only_batch_norm=True)
        self._pool1 = tf.keras.layers.MaxPool2D(
            pool_size=2, strides=2, padding="same")
        self._dropout1 = tf.keras.layers.Dropout(0.5)

        self._conv2a = Conv2D(filters=256,kernel_size=[3,3], 
                                                        padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                                        kernel_initializer=tf.keras.initializers.he_uniform(),
                                                        bias_initializer=tf.keras.initializers.constant(
                                                            0.1),
                                                        weight_norm=True, mean_only_batch_norm=True)
        self._conv2b = Conv2D(filters=256,kernel_size=[3,3],
                                                        padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                                        kernel_initializer=tf.keras.initializers.he_uniform(),
                                                        bias_initializer=tf.keras.initializers.constant(
                                                            0.1),
                                                        weight_norm=True, mean_only_batch_norm=True)
        self._conv2c = Conv2D(filters=256,kernel_size=[3,3],
                                                        padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                                        kernel_initializer=tf.keras.initializers.he_uniform(),
                                                        bias_initializer=tf.keras.initializers.constant(
                                                            0.1),
                                                        weight_norm=True, mean_only_batch_norm=True)
        self._pool2 = tf.keras.layers.MaxPool2D(
            pool_size=2, strides=2, padding="same")
        self._dropout2 = tf.keras.layers.Dropout(0.5)

        self._conv3a = Conv2D(filters=512,kernel_size=[3,3],padding="valid", activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                                        kernel_initializer=tf.keras.initializers.he_uniform(),
                                                        bias_initializer=tf.keras.initializers.constant(
                                                            0.1),
                                                        weight_norm=True, mean_only_batch_norm=True)
        self._conv3b = Conv2D(filters=256,kernel_size=[3,3], 
                                                        padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                                        kernel_initializer=tf.keras.initializers.he_uniform(),
                                                        bias_initializer=tf.keras.initializers.constant(
                                                            0.1),
                                                        weight_norm=True, mean_only_batch_norm=True)
        self._conv3c = Conv2D(filters=128,kernel_size=[3,3], 
                                                        padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                                        kernel_initializer=tf.keras.initializers.he_uniform(),
                                                        bias_initializer=tf.keras.initializers.constant(
                                                            0.1),
                                                        weight_norm=True, mean_only_batch_norm=True)

        self._dense = Dense(units=2, activation=tf.nn.softmax,
                                                     kernel_initializer=tf.keras.initializers.he_uniform(),
                                                     bias_initializer=tf.keras.initializers.constant(
                                                         0.1),
                                                     weight_norm=True, mean_only_batch_norm=True)

    def __aditive_gaussian_noise(self, input, std):
        """ Function to add additive zero mean noise as described in the paper

        Arguments:
            input {tensor} -- image
            std {int} -- std to use in the random_normal

        Returns:
            {tensor} -- image with added noise
        """

        noise = tf.random.normal(shape=tf.shape(
            input), mean=0.0, stddev=std, dtype=tf.float32)
        return input + noise

    def __apply_image_augmentation(self, image):
        """ Applies random transformation to the image (shift image withina range of 
            [-2, 2] pixels)

        Arguments:
            image {tensor} -- image

        Returns:
            {tensor} -- transformed image
        """

        random_shifts = np.random.randint(-2, 2, (image.numpy().shape[0], 2))
        random_transformations = tf.contrib.image.translations_to_projective_transforms(
            random_shifts)
        image = tf.contrib.image.transform(image, random_transformations, 'NEAREST',
                                           output_shape=tf.convert_to_tensor(image.numpy().shape[1:3], dtype=np.int32))
        return image

    def call(self, input, training=True):
        """ Function that allows running a tensor through the pi model

        Arguments:
            input {[tensor]} -- batch of images
            training {bool} -- if true applies augmentaton and additive noise

        Returns:
            [tensor] -- predictions"""

 

        # if training:
        #     h = self.__aditive_gaussian_noise(input, 0.15)
        #     h = self.__apply_image_augmentation(h)
        # else:
        h = input
        h=self._emb(h)
        
        h= create_embedding(h)

        h = self._conv1a(h, training)
        h = self._conv1b(h, training)
        h = self._conv1c(h, training)
        h = self._pool1(h)
        h = self._dropout1(h, training=training)

        h = self._conv2a(h, training)
        h = self._conv2b(h, training)
        h = self._conv2c(h, training)
        h = self._pool2(h)
        h = self._dropout2(h, training=training)

        h = self._conv3a(h, training)
        h = self._conv3b(h, training)
        h = self._conv3c(h, training)

        # Average Pooling
        h = tf.compat.v1.reduce_mean(h, reduction_indices=[1, 2])
        return self._dense(h, training)
