import tensorflow as tf
import numpy as np

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(84, 84, 3)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=5, strides=(3, 3), activation=tf.nn.relu),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=5, strides=(2, 2), activation=tf.nn.relu),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=5, strides=(2, 2), activation=tf.nn.relu),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
            ])
        low_dim = 7
        self.generative_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=low_dim*low_dim*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(low_dim, low_dim, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=5,
                strides=(3, 3),
                padding="SAME",
                activation=tf.nn.relu),
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=5,
                strides=(2, 2),
                padding="SAME",
                activation=tf.nn.relu),
            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=5,
                strides=(2, 2),
                padding="SAME",
                activation=tf.nn.relu),
            # No activation
            tf.keras.layers.Conv2DTranspose(
              filters=3, kernel_size=3, strides=(1, 1), padding="SAME"),
            ])

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random_normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random_normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
          probs = tf.sigmoid(logits)
          return probs

        return logits

    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

class SpatialSoftMax(tf.keras.layers.Layer):
    # def __init__(self):
    #     super(SpatialSoftMax, self).__init__()

    # def build(self, input_shape):
    #     self.kernel

    def call(self, inputs):
        return tf.contrib.layers.spatial_softmax(inputs)

    # def compute_output_shape(self, input_shape):


class SpatialAE(tf.keras.Model):
    def __init__(self, latent_dim, dropout=0.0):
        super(SpatialAE, self).__init__()
        self.latent_dim = latent_dim
        self.dropout = dropout
        assert self.latent_dim%2 == 0

        self.inference_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(84, 84, 3)),
            tf.keras.layers.Conv2D(
                filters=32, 
                kernel_size=5, 
                strides=(3, 3), 
                activation=tf.nn.relu),
            tf.keras.layers.Conv2D(
                filters=64, 
                kernel_size=5, 
                strides=(2, 2), 
                activation=tf.nn.relu),
            tf.keras.layers.Conv2D(
                filters=int(latent_dim/2), 
                kernel_size=5, 
                strides=(2, 2), 
                activation=tf.nn.relu),
            SpatialSoftMax(),
            # tf.keras.layers.Flatten(),
            # # No activation
            # tf.keras.layers.Dense(latent_dim),
            tf.keras.layers.Dropout(self.dropout),
            ])

        # self.inference_net.add(
        #     tf.keras.layers.InputLayer(
        #         input_tensor=tf.contrib.layers.spatial_softmax(
        #             self.inference_net.layers[-1].output,
        #             trainable=False
        #             )
        #         )
        #     )
        # self.inference_net.add(
        #     tf.keras.layers.Flatten()
        #     )
        # self.inference_net.add(
        #     tf.keras.layers.Dense(latent_dim)            
        #     )
        #self.inference_net = self.inference_net()        
        low_dim = 7
        
        self.generative_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=low_dim*low_dim*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(low_dim, low_dim, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=5,
                strides=(3, 3),
                padding="SAME",
                activation=tf.nn.relu),
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=5,
                strides=(2, 2),
                padding="SAME",
                activation=tf.nn.relu),
            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=5,
                strides=(2, 2),
                padding="SAME",
                activation=tf.nn.relu),
            # No activation
            tf.keras.layers.Conv2DTranspose(
              filters=3, kernel_size=3, strides=(1, 1), padding="SAME"),
            ])

    def compute_loss(self, x):
        features = self.inference_net(x)
        reconstruction  = self.generative_net(features)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstruction, labels=x)
        return tf.reduce_mean(cross_ent)