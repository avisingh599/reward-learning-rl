import os
import os.path as osp
import time
import glob

import matplotlib.pyplot as plt
import PIL
import imageio
import numpy as np
import tensorflow as tf

#from softlearning.autoencoder.autoencoder_tf import VAE, SpatialAE
from softlearning.models.autoencoder_models import spatialAE

HDD = '/root/softlearning/data/'
ENV_NAME = 'sawyer_pusher_no_texture'
data_directory_experts = '/root/gym-larry/gym/envs/mujoco/assets/sawyer_pusher_data/expert_images_randomize_gripper_False_pos_noise_0.01_texture_False/task40/'
model_save_path = '/root/ray_results/autoencoder_models_tf/'

def load_data():

    data_directory = osp.join(HDD, 'random_trajectories', ENV_NAME)
    #data_path = osp.join(save_directory, 'combined_images.pkl')
    images = []
    file_list = sorted(os.listdir(data_directory_experts))
    for fname in file_list:
        if fname.endswith('.png'):
            image_path = os.path.join(data_directory_experts, fname)
            image = imageio.imread(image_path)
            image = image.astype(np.float32) / 255.
            images.append(image)
    file_list =  sorted(os.listdir(data_directory))
    for fname in file_list:
        if fname.endswith('.png'):
            image_path = os.path.join(data_directory, fname)
            image = imageio.imread(image_path)
            image = image.astype(np.float32) / 255.
            images.append(image)

    images = np.array(images)

    return images

def main():

    from tensorflow.keras.backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                        # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras    

    latent_dim = 32
    model = spatialAE(latent_dim)
    model.compile(optimizer='adam',
        loss={'reconstruction': 'mean_squared_error'})
    images = load_data()
    model.fit(images, images, epochs=100, batch_size=128, validation_split=0.1)
    model.save_weights(osp.join(model_save_path, 'spatial_ae.h5'))
    #import IPython; IPython.embed()

if __name__ == "__main__":
    main()