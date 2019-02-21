"""Collect data using random actions."""

import os
import os.path as osp
import pickle
#from rllab.policies.uniform_control_policy import UniformControlPolicy
#from sac.envs.normalized_env import normalize
#from sac.autoencoder.autoencoder_collect import collect_data
#from gym.envs.mujoco.multitask.sawyer_pusher_env import SawyerPushXYEasyEnv
from gym.envs.mujoco.multitask.sawyer_pusher_multienv import SawyerPushXYMultiEnv
from softlearning.policies.utils import get_policy_from_variant, get_policy
from softlearning.samplers import rollouts

from softlearning.environments.adapters.gym_adapter import GymAdapterPixel

NUM_TRAJECTORIES = 100
MAX_PATH_LENGTH = 150
# ENV_NAME = 'sawyer_pusher_texture'
ENV_NAME = 'sawyer_pusher_no_texture'

import imageio

def main():
    """Collect and save images from a random policy. """

    #env = normalize(SawyerPushXYMultiEnv(40, hide_goal=True, texture=True))
    env_ = GymAdapterPixel(env=SawyerPushXYMultiEnv(task_id=40, hide_goal=False, texture=False))

    policy = get_policy('UniformPolicy', env_)
    #print('Collecting trajectories...')
    #import IPython; IPython.embed()


    # training_data = collect_data(
    #     env=env,
    #     agent=agent,
    #     num_trajectories=NUM_TRAJECTORIES,
    #     max_path_length=MAX_PATH_LENGTH,
    # )

#    print('Finished collecting trajectories.')
    
    save_directory = osp.join('/root/softlearning/data/random_trajectories', ENV_NAME)
    if not osp.exists(save_directory):
        os.makedirs(save_directory)

    print('Saving images')
    for i in range(NUM_TRAJECTORIES):
        print(i)
        path = rollouts(env_,
                 policy,
                 path_length=MAX_PATH_LENGTH,
                 n_paths=1,
                 render_mode=None)
        path = path[0]
        for j in range(path['observations'].shape[0]):
            image_save_name = '{}/{}_{}.png'.format(save_directory, i, j)
            imageio.imwrite(image_save_name, path['observations'][j].reshape(84, 84, 3))
    #save_path = osp.join(save_directory, 'data.pkl')
    #print('Saving data to: {}'.format(save_path))
    #with open(save_path, 'wb') as file:
    #    pickle.dump(training_data, file)
    #print('Finished saving data to: {}'.format(save_path))


if __name__ == "__main__":
    main()
