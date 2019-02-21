import os
import numpy as np
from gym.envs.mujoco.multitask.sawyer_pusher_multienv import SawyerPushXYMultiEnv

import matplotlib.pyplot as plt
#from sac.policies.uniform_policy import UniformPolicy
#from sac.policies.constant_policy import ConstantPolicy
#from sac.envs.autoencoder_env import normalize_autoencoder
from softlearning.environments.adapters.gym_adapter import GymAdapter,\
    GymAdapterAutoEncoder
from softlearning.autoencoder.autoencoder import AE

#from tqdm import trange
#import scipy
import imageio
import argparse

def get_expert_data(env, randomize_gripper, pos_noise):
    env.reset()
    #import IPython; IPython.embed()
    goal_pos = env._env.env.get_goal_pos()[:2]

    if randomize_gripper:
        ARM_LOW, ARM_HIGH = np.array([-0.08, 0.40]), np.array([0.08, 0.75])
        arm_pos = np.random.uniform(low=ARM_LOW, high=ARM_HIGH)
    else:
        arm_pos = goal_pos - np.array([0, 0.075])
        arm_pos += np.random.uniform([-0.03, -0.075], [0.03, 0])

    block_pos = goal_pos + np.random.uniform([-pos_noise, -pos_noise], [pos_noise, pos_noise])

    env._env.env.move_to(arm_pos)
    env._env.env.set_block_xy(block_pos)

    image = env._env.env._get_image(84, 84, 'robotview')
    full_state = env._env.env._get_obs()
    joint_state = env._env.env.get_endeff_pos()[:2]
    #joint_state = full_state[-env._joint_dim:]

    return image, joint_state, full_state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_experts", type=int, default=250,
                        help="number of experts collected per task")
    parser.add_argument("--pos_noise", type=float, default=0.01,
                        help="noise added to the block position")
    parser.add_argument("--task_start", type=int, default=40,
                        help="starting task ind")
    parser.add_argument("--task_end", type=int, default=41,
                        help="ending task ind")
    parser.add_argument("--texture", action="store_true",
                        help="ending task ind")
    parser.add_argument("--randomize_gripper", action="store_true",
                        help="ending task ind")
    args = parser.parse_args()

    # autoencoder_savepath='/root/sac-plus/autoencoder/sawyer_pusher_meta/ae.model'
    SAVE_DIR='/root/gym-larry/gym/envs/mujoco/assets/sawyer_pusher_data/expert_images'
    SAVE_DIR+='_randomize_gripper_{}'.format(args.randomize_gripper)
    SAVE_DIR+='_pos_noise_{}'.format(args.pos_noise)
    SAVE_DIR+='_texture_{}'.format(args.texture)
    SAVE_DIR+='/'

    tasks_to_target = list(range(args.task_start, args.task_end))

    for task_id in tasks_to_target:
        task_dir = os.path.join(SAVE_DIR, 'task{}'.format(task_id))
        if not os.path.isdir(task_dir):
            os.makedirs(task_dir)

        # env = normalize_autoencoder(SawyerPushXYMultiEnv(task_id, hide_goal=True, texture=args.texture),
        #                         autoencoder_savepath=autoencoder_savepath,
        #                         use_jointstate=True)
        #env._env.env.set_task_id(task_id)
        if args.texture:
            hide_goal=True
        else:
            hide_goal=False

        env = GymAdapterAutoEncoder(
            env=SawyerPushXYMultiEnv(
                task_id=40, 
                hide_goal=hide_goal,
                texture=args.texture,
                pos_noise=0.0,
                randomize_gripper=False,
                forward_only=False,
                ),
            autoencoder_model=AE(),
            autoencoder_savepath='/root/softlearning/data/'
            'autoencoder_models/sawyer_pusher_texture/ae_better.pwf'
            )

        for expert_idx in range(args.num_experts):
            image, joint_state, full_state = get_expert_data(
                env, args.randomize_gripper, args.pos_noise)
            image_path = os.path.join(task_dir, 'image_{}.png'.format(expert_idx))
            joint_path = os.path.join(task_dir, 'endeff_{}.txt'.format(expert_idx))
            fullstate_path = os.path.join(task_dir, 'fullstate_{}.txt'.format(expert_idx))
            imageio.imsave(image_path, image)
            np.savetxt(joint_path, joint_state, delimiter=',')
            np.savetxt(fullstate_path, full_state, delimiter=',')

if __name__ == '__main__':
    main()
