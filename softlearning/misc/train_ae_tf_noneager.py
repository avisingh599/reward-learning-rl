import os
import os.path as osp
import datetime

import imageio
import numpy as np
import tensorflow as tf

#from softlearning.autoencoder.autoencoder_tf import VAE, SpatialAE
from softlearning.models.autoencoder_models import spatialAE

HDD = '/root/softlearning/data/'
ENV_NAME = 'sawyer_pusher_no_texture'
data_directory_experts = '/root/gym-larry/gym/envs/mujoco/assets/sawyer_pusher_data' \
+'/expert_images_randomize_gripper_False_pos_noise_0.01_texture_False/task40/'
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
    #limit memory allocation
    from tensorflow.keras.backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.Session(config=config)
    set_session(sess)

    latent_dim = 32
    model = spatialAE(latent_dim)
    model.compile(optimizer='adam',
        loss={'reconstruction': 'mean_squared_error'})
    images = load_data()

    experiment_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = osp.join(model_save_path, experiment_id)
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