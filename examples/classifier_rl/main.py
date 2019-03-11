import os
import copy
import glob
import pickle
import sys

import tensorflow as tf
import numpy as np
from ray import tune

from softlearning.environments.utils import get_environment_from_variant
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import get_policy_from_variant, get_policy
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant
from softlearning.models.utils import get_reward_classifier_from_variant

from softlearning.misc.utils import set_seed, initialize_tf_variables
from examples.instrument import run_example_local
from examples.development.main import ExperimentRunner

from softlearning.environments.adapters.gym_adapter import GymAdapter,\
    GymAdapterAutoEncoderTF, GymAdapterPixel
from gym.envs.mujoco.multitask.sawyer_pusher_multienv import \
    SawyerPushXYMultiEnv

#TODO Avi move this to a get_env function
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import \
    SawyerPushAndReachXYEnv, SawyerPushAndReachXYZEnv
from multiworld.core.image_env import ImageEnv
from multiworld.core.flat_goal_env import FlatGoalEnv
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2


#TODO Avi move this somewhere else
ae_address = {
    'texture': {
        'spatial_ae': ('/root/ray_results/autoencoder_models_tf/'
            '2019-02-27_23-14-05_num_expert_images-200_env_type-sawyer_pusher_texture/'
            'spatial_ae.h5'),
        'vanilla_ae': ('/root/ray_results/autoencoder_models_tf/'
            '2019-03-01_04-31-15_autoencoder_type-vanilla_ae_num_expert_images-200'
            '_env_type-sawyer_pusher_texture/model.h5')

    },

    'no-texture': {
        'spatial_ae': ('/root/ray_results/autoencoder_models_tf/'
            '2019-02-28_00-48-40_num_expert_images-10_env_type-sawyer_pusher_no_texture/'
            'spatial_ae.h5'),

    }
}

class ExperimentRunnerClassifierRL(ExperimentRunner):

    def _build(self):
        variant = copy.deepcopy(self._variant)

        #TODO Avi Implement a new version of get_env_from_variant
        #also rename paths according to this scheme
        #from softlearning.misc.utils import PROJECT_PATH 
        #ae_path = os.path.join(PROJECT_PATH, 'autoencoder_models', 'sawyer_pusher_no_texture', 'vae.pwf')

        if variant['texture']:
            hide_goal = True
            ae_path = ae_address['texture'][variant['autoencoder_type']]
        else:
            hide_goal = False
            ae_path = ae_address['no-texture'][variant['autoencoder_type']]

        goal_vec = {
            #'state_desired_goal': np.asarray([0.0, 0.7, 0.02, 0.0, 0.8])
            'state_desired_goal': np.asarray([0.0, 0.6, 0.02, -0.15, 0.6])   
        }

        if variant['perception'] == 'autoencoder':
            env = self.env = GymAdapterAutoEncoderTF(
                autoencoder_savepath=ae_path,
                env=SawyerPushXYMultiEnv(
                    task_id=40, 
                    hide_goal=hide_goal,
                    texture=variant['texture'],
                    pos_noise=0.01,
                    randomize_gripper=False,
                    forward_only=False,
                    ),
                )

        elif variant['perception'] == 'full_state':
            env = SawyerPushAndReachXYEnv(
                    fix_goal=True,
                    reward_type='puck_distance',
                    fixed_goal=goal_vec['state_desired_goal'],
                    puck_radius=.05,
                    )
            env_flat = FlatGoalEnv(env, obs_keys=['observation'])
            env = self.env = GymAdapter(env=env_flat)

        elif variant['perception'] == 'pixel':
            env = ImageEnv(
                    SawyerPushAndReachXYEnv(
                        fix_goal=True,
                        reward_type='puck_distance',
                        fixed_goal=goal_vec['state_desired_goal'],
                        puck_radius=.05,
                        ),
                    init_camera=sawyer_pusher_camera_upright_v2,
                    normalize=True,
                    )
            env_flat = FlatGoalEnv(env, obs_keys=['image_observation'])
            env = self.env = GymAdapter(env=env_flat)
        else:
            raise NotImplementedError

        replay_pool = self.replay_pool = (
            get_replay_pool_from_variant(variant, env))
        sampler = self.sampler = get_sampler_from_variant(variant)
        Qs = self.Qs = get_Q_function_from_variant(variant, env)
        policy = self.policy = get_policy_from_variant(variant, env, Qs)
        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy('UniformPolicy', env))

        algorithm_kwargs = {
            'variant': self._variant,
            'env': self.env,
            'policy': policy,
            'initial_exploration_policy': initial_exploration_policy,
            'Qs': Qs,
            'pool': replay_pool,
            'sampler': sampler,
            'session': self._session,
        }

        if self._variant['algorithm_params']['type'] in ['SACClassifier', 'RAQ', 'VICE', 'VICERAQ']:
            reward_classifier = self.reward_classifier \
                = get_reward_classifier_from_variant(self._variant, env)
            algorithm_kwargs['classifier'] = reward_classifier

            #TODO Avi maybe write a "get_data_from_variant"
            # if self._variant['perception'] == 'autoencoder':
            #     goal_images = env._env.env.get_expert_images()
            #     goal_aefeatures = env.feature_points(goal_images)
            #     goal_examples = goal_aefeatures
            # elif self._variant['perception'] == 'full_state':
            #     goal_examples = env._env.env.get_expert_fullstates()
            # elif self._variant['perception'] == 'pixel':
            #     goal_images = env._env.env.get_expert_images()
            #     n_images = goal_images.shape[0]
            #     goal_examples = goal_images.reshape((n_images, -1))
            # else:
            #     raise NotImplementedError

            #n_goal_examples = self._variant['data_params']['n_goal_examples']
            #assert goal_examples.shape[0] >= n_goal_examples, goal_examples.shape


            goal_examples = []
            total_goal_examples = self._variant['data_params']['n_goal_examples'] \
                + self._variant['data_params']['n_goal_examples_validation_max']
            
            for i in range(total_goal_examples):
                env._env.env.reset()
                goal_vec['state_desired_goal'] += np.random.uniform(low=-0.01, high=0.01, size=(5,))
                env._env.env.set_to_goal(goal_vec)
                ob, rew, done, info = env_flat.step(np.asarray([0.,0.]))
                goal_examples.append(ob)
            goal_examples = np.asarray(goal_examples)
            env._env.env.reset()
            n_goal_examples = self._variant['data_params']['n_goal_examples']
            algorithm_kwargs['goal_examples'] = goal_examples[:n_goal_examples]
            algorithm_kwargs['goal_examples_validation'] = \
                goal_examples[n_goal_examples:]

        self.algorithm = get_algorithm_from_variant(**algorithm_kwargs)

        initialize_tf_variables(self._session, only_uninitialized=True)

        self._built = True

    def _restore(self, checkpoint_dir):
        assert isinstance(checkpoint_dir, str), checkpoint_dir

        checkpoint_dir = checkpoint_dir.rstrip('/')

        with self._session.as_default():
            pickle_path = self._pickle_path(checkpoint_dir)
            with open(pickle_path, 'rb') as f:
                picklable = pickle.load(f)

        env = self.env = picklable['env']

        replay_pool = self.replay_pool = (
            get_replay_pool_from_variant(self._variant, env))

        if self._variant['run_params'].get('checkpoint_replay_pool', False):
            self._restore_replay_pool(checkpoint_dir)

        sampler = self.sampler = picklable['sampler']
        Qs = self.Qs = picklable['Qs']
        # policy = self.policy = picklable['policy']
        policy = self.policy = (
            get_policy_from_variant(self._variant, env, Qs))
        self.policy.set_weights(picklable['policy_weights'])
        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy('UniformPolicy', env))

        algorithm_kwargs = {
            'variant': self._variant,
            'env': self.env,
            'policy': policy,
            'initial_exploration_policy': initial_exploration_policy,
            'Qs': Qs,
            'pool': replay_pool,
            'sampler': sampler,
            'session': self._session,
        }

        if self._variant['algorithm_params']['type'] in ['SACClassifier', 'RAQ', 'VICE', 'VICERAQ']:
            reward_classifier = self.reward_classifier = picklable['reward_classifier']
            algorithm_kwargs['classifier'] = reward_classifier

            #TODO Avi maybe write a "get_data_from_variant"
            if self._variant['perception'] == 'autoencoder':
                goal_images = env._env.env.get_expert_images()
                goal_aefeatures = env.feature_points(goal_images)
                goal_examples = goal_aefeatures
            elif self._variant['perception'] == 'full_state':
                goal_examples = env._env.env.get_expert_fullstates()
            elif self._variant['perception'] == 'pixel':
                goal_images = env._env.env.get_expert_images()
                n_images = goal_images.shape[0]
                goal_examples = goal_images.reshape((n_images, -1))
            else:
                raise NotImplementedError

            n_goal_examples = self._variant['data_params']['n_goal_examples']
            assert goal_examples.shape[0] >= n_goal_examples

            n_goal_examples_validation_max = self._variant['data_params']['n_'
                        'goal_examples_validation_max']
            algorithm_kwargs['goal_examples'] = goal_examples[:n_goal_examples]
            algorithm_kwargs['goal_examples_validation'] = \
                goal_examples[n_goal_examples:n_goal_examples_validation_max+n_goal_examples]

        self.algorithm = get_algorithm_from_variant(**algorithm_kwargs)
        self.algorithm.__setstate__(picklable['algorithm'].__getstate__())

        tf_checkpoint = self._get_tf_checkpoint()
        status = tf_checkpoint.restore(tf.train.latest_checkpoint(
            os.path.split(self._tf_checkpoint_prefix(checkpoint_dir))[0]))

        status.assert_consumed().run_restore_ops(self._session)
        initialize_tf_variables(self._session, only_uninitialized=True)

        # TODO(hartikainen): target Qs should either be checkpointed or pickled.
        for Q, Q_target in zip(self.algorithm._Qs, self.algorithm._Q_targets):
            Q_target.set_weights(Q.get_weights())

        self._built = True

    @property
    def picklables(self):
        picklables = {
            'variant': self._variant,
            'env': self.env,
            'sampler': self.sampler,
            'algorithm': self.algorithm,
            'Qs': self.Qs,
            'policy_weights': self.policy.get_weights(),
        }

        if hasattr(self, 'reward_classifier'): 
            picklables['reward_classifier'] = self.reward_classifier

        return picklables

def main(argv=None):
    """Run ExperimentRunner locally on ray.

    To run this example on cloud (e.g. gce/ec2), use the setup scripts:
    'softlearning launch_example_{gce,ec2} examples.development <options>'.

    Run 'softlearning launch_example_{gce,ec2} --help' for further
    instructions.
    """
    # __package__ should be `development.main`
    run_example_local('classifier_rl.main', argv)


if __name__ == '__main__':
    main(argv=sys.argv[1:])
