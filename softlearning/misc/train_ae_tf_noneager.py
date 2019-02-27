import os
import os.path as osp
import datetime
import argparse

import imageio
import numpy as np
import tensorflow as tf

#from softlearning.autoencoder.autoencoder_tf import VAE, SpatialAE
from softlearning.models.autoencoder_models import spatialAE

HDD = '/root/softlearning/data/'
data_directory_experts = { 
    'sawyer_pusher_no_texture': '/root/gym-larry/gym/envs/mujoco/assets/sawyer_pusher_data' \
    +'/expert_images_randomize_gripper_False_pos_noise_0.01_texture_False/task40/',
    'sawyer_pusher_texture': '/root/gym-larry/gym/envs/mujoco/assets/sawyer_pusher_data' \
    +'/expert_images_randomize_gripper_False_pos_noise_0.01_texture_True/task40/'
}
model_save_path = '/root/ray_results/autoencoder_models_tf/'

def load_data(n_expert_images, env_type):

    data_directory = osp.join(HDD, 'random_trajectories', env_type)
    #data_path = osp.join(save_directory, 'combined_images.pkl')
    images = []
    file_list = sorted(os.listdir(data_directory_experts[env_type]))
    for fname in file_list:
        if fname.endswith('.png'):
            image_path = os.path.join(data_directory_experts[env_type], fname)
            image = imageio.imread(image_path)
            image = image.astype(np.float32) / 255.
            images.append(image)

    assert len(images) >= n_expert_images, len(images)
    images = images[:n_expert_images]

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_expert_images', type=int, default=200)
    parser.add_argument('--env_type', type=str, default='sawyer_pusher_no_texture',
        choices=('sawyer_pusher_no_texture', 'sawyer_pusher_texture'))
    args = parser.parse_args()

    experiment_id= datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_id+= '_num_expert_images-{}'.format(args.n_expert_images)
    experiment_id+= '_env_type-{}'.format(args.env_type)
    log_dir = osp.join(model_save_path, experiment_id)

    #limit initial GPU memory allocation
    from tensorflow.keras.backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.Session(config=config)
    set_session(sess)

    images = load_data(args.n_expert_images, args.env_type)

    latent_dim = 32
    model = spatialAE(latent_dim)
    model.compile(optimizer='adam',
        loss={'reconstruction': 'mean_squared_error'})

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    checkpointCallBack = tf.keras.callbacks.ModelCheckpoint(
        osp.join(log_dir, 'spatial_ae.h5'), monitor='reconstruction_loss', verbose=1, 
        save_best_only=True, save_weights_only=True, mode='min')
    tbCallBack = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=0, write_graph=True, write_images=True)
    model.fit(images, images, epochs=100,
        batch_size=128, validation_split=0.1,
        callbacks=[tbCallBack, checkpointCallBack])
    #model.save_weights(osp.join(log_dir, 'spatial_ae.h5'))

if __name__ == "__main__":
    main()