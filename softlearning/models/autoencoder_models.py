import numpy as np
import tensorflow as tf

from softlearning.utils.keras import PicklableKerasModel


def spatial_ae(latent_dim):
    """
    Implements the Deep Spatial AutoEncoder described in Finn et al. (2016)
    """
    assert latent_dim%2 == 0, latent_dim
    input_image = tf.keras.layers.Input(shape=(84, 84, 3))

    conv = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=5,
        strides=(3, 3),
        activation=tf.nn.relu)(input_image)
    conv = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=5,
        strides=(3, 3),
        activation=tf.nn.relu)(conv)
    conv = tf.keras.layers.Conv2D(
        filters=int(latent_dim/2),
        kernel_size=5,
        strides=(3, 3),
        activation=tf.nn.relu)(conv)

    #feature_points = tf.contrib.layers.spatial_softmax(conv, name='spatial_softmax')
    feature_points = SpatialSoftMax()(conv)
    feature_points_dropout = tf.keras.layers.Dropout(0.5)(feature_points)

    low_dim = 7 #image dimension of downsampled image
    out = tf.keras.layers.Dense(units=low_dim*low_dim*32,
        activation=tf.nn.relu)(feature_points_dropout)
    out = tf.keras.layers.Reshape(target_shape=(low_dim, low_dim, 32))(out)
    out = tf.keras.layers.Conv2DTranspose(
        filters=64,
        kernel_size=5,
        strides=(3, 3),
        padding="SAME",
        activation=tf.nn.relu)(out)
    out = tf.keras.layers.Conv2DTranspose(
        filters=64,
        kernel_size=5,
        strides=(2, 2),
        padding="SAME",
        activation=tf.nn.relu)(out)
    out = tf.keras.layers.Conv2DTranspose(
        filters=32,
        kernel_size=5,
        strides=(2, 2),
        padding="SAME",
        activation=tf.nn.relu)(out)
    # No activation
    reconstruction = tf.keras.layers.Conv2DTranspose(
              filters=3, kernel_size=3, strides=(1, 1), padding="SAME", name='reconstruction')(out)

    return PicklableKerasModel(inputs=input_image, outputs=[feature_points, reconstruction])


class SpatialSoftMax(tf.keras.layers.Layer):
    """
    Implements the spatialSoftMax layer from Levine*, Finn* et al. (2016)
    """
    def call(self, inputs):
        #implementation from tf.contrib.layers.spatial_softmax
        #the follwoing line does not work because of a pesky
        #temperature variable creation
        #return tf.contrib.layers.spatial_softmax(inputs, temperature=1.0, trainable=True)

        #TODO Avi maybe add temperature here 
        #softmax_attention = nn.softmax(features / temperature)
        #import IPython; IPython.embed()
        shape = tf.shape(inputs)
        static_shape = inputs.shape
        height, width, num_channels = shape[1], shape[2], static_shape[3]
        pos_x, pos_y = tf.meshgrid(
            tf.lin_space(-1., 1., num=height),
            tf.lin_space(-1., 1., num=width),
            indexing='ij')
        pos_x = tf.reshape(pos_x, [height * width])
        pos_y = tf.reshape(pos_y, [height * width])

        inputs = tf.reshape(
            tf.transpose(inputs, [0, 3, 1, 2]), [-1, height * width])

        softmax_attention = tf.nn.softmax(inputs)
        expected_x = tf.reduce_sum(pos_x * softmax_attention, [1], keepdims=True)
        expected_y = tf.reduce_sum(pos_y * softmax_attention, [1], keepdims=True)
        expected_xy = tf.concat([expected_x, expected_y], 1)
        feature_keypoints = tf.reshape(expected_xy,
                                            [-1, num_channels.value * 2])
        feature_keypoints.set_shape([None, num_channels.value * 2])
        return feature_keypoints

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[3])
