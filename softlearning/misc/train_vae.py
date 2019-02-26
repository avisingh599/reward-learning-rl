import os
import os.path as osp
import pickle
from gym.envs.mujoco.multitask.sawyer_pusher_env import SawyerPushXYEasyEnv
from softlearning.autoencoder.autoencoder_trainer import AutoencoderTrainer
from softlearning.autoencoder.autoencoder import AE, VAE
import numpy as np
import imageio

ENV_NAME = 'sawyer_pusher_no_texture'
MODEL = VAE
AUTOENCODER_TYPE = MODEL.__name__.lower()
HDD = '/root/softlearning/data/'

def main():

    #import IPython; IPython.embed()
    save_directory = osp.join(HDD, 'autoencoder_models', ENV_NAME)
    test_directory = osp.join(save_directory, 'test_{}'.format(AUTOENCODER_TYPE))
    if not osp.isdir(test_directory):
        os.makedirs(test_directory)

    data_directory = osp.join(HDD, 'random_trajectories', ENV_NAME)
    #data_path = osp.join(save_directory, 'combined_images.pkl')
    images = []
    for fname in sorted(os.listdir(data_directory)):
        if fname.endswith('.png'):
            image_path = os.path.join(data_directory, fname)
            image = imageio.imread(image_path)
            image = image.astype(np.float32) / 255.
            images.append(image)

    data_directory_experts = '/root/gym-larry/gym/envs/mujoco/assets/sawyer_pusher_data/'
    'expert_images_randomize_gripper_False_pos_noise_0.01_texture_False/task40/'
    for fname in sorted(os.listdir(data_directory_experts)):
        if fname.endswith('.png'):
            image_path = os.path.join(data_directory_experts, fname)
            image = imageio.imread(image_path)
            image = image.astype(np.float32) / 255.
            images.append(image)
    
    images = np.asarray(images)

    save_path = osp.join(save_directory, '{}.pwf'.format(AUTOENCODER_TYPE))

    # with open(data_path, 'rb') as file:
    #     data = pickle.load(file)


    #import ipdb; ipdb.set_trace();
    train_data, test_data = AutoencoderTrainer.split_dataset(images, 0.95)

    #model = AE().cuda()
    model = VAE(num_dims=4).cuda()
    trainer = AutoencoderTrainer(model, train_data, test_data, test_directory, save_path)
    trainer.train(num_epochs=1000, n_val=10)
    trainer.save(save_path)


if __name__ == "__main__":
    main()
