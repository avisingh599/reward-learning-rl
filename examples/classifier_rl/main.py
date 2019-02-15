import os
import copy
import glob
import pickle
import sys

import tensorflow as tf
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
    GymAdapterAutoEncoder
from gym.envs.mujoco.multitask.sawyer_pusher_multienv import \
    SawyerPushXYMultiEnv
from softlearning.autoencoder.autoencoder import AE

class ExperimentRunnerClassifierRL(ExperimentRunner):

    def _build(self):
        variant = copy.deepcopy(self._variant)

        #env = self.env = get_environment_from_variant(variant)
        #env = self.env = GymAdapter(env=SawyerPushXYEnv())
        #'/root/sac-plus/experiments/autoencoder/sawyer_pusher_texture/ae.pwf',

        env = self.env = GymAdapterAutoEncoder(
            env=SawyerPushXYMultiEnv(
                task_id=40, 
                hide_goal=True,
                texture=True,
                pos_noise=0.0,
                randomize_gripper=False,
                forward_only=False,
                ),
            autoencoder_model=AE(),
            autoencoder_savepath='/root/softlearning/data/'
            'autoencoder_models/sawyer_pusher_texture/ae_better.pwf'
            )
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
            reward_classifier = get_reward_classifier_from_variant(self._variant, env)
            algorithm_kwargs['classifier'] = reward_classifier

            goal_images = env._env.env.get_expert_images()
            goal_aefeatures = env.feature_points(goal_images)
            goal_examples = goal_aefeatures

            algorithm_kwargs['goal_examples'] = goal_examples

        self.algorithm = get_algorithm_from_variant(**algorithm_kwargs)

        initialize_tf_variables(self._session, only_uninitialized=True)

        self._built = True


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
