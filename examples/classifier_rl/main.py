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
    GymAdapterAutoEncoder, GymAdapterAutoEncoderTF
from gym.envs.mujoco.multitask.sawyer_pusher_multienv import \
    SawyerPushXYMultiEnv
from softlearning.autoencoder.autoencoder import AE, VAE
from softlearning.models.autoencoder_models import spatialAE

class ExperimentRunnerClassifierRL(ExperimentRunner):

    def _build(self):
        variant = copy.deepcopy(self._variant)

        #env = self.env = get_environment_from_variant(variant)
        #env = self.env = GymAdapter(env=SawyerPushXYEnv())
        #'/root/sac-plus/experiments/autoencoder/sawyer_pusher_texture/ae.pwf',

        #TODO Avi Implement a new version of get_env_from_variant
        if variant['perception'] == 'autoencoder':
            if variant['texture']:
                hide_goal = True
                ae_path = '/root/softlearning/data/' \
                + 'autoencoder_models/sawyer_pusher_texture/ae_better.pwf'
                ae_model = AE()
            else:
                hide_goal = False
                ae_path = '/root/softlearning/data/' \
                + 'autoencoder_models_tf/spatial_ae.h5'
                latent_dim = 32
                ae_model = spatialAE(latent_dim)

            #import IPython; IPython.embed()
            env = self.env = GymAdapterAutoEncoderTF(
                autoencoder_model=ae_model,
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
            env = self.env = GymAdapter(
                env=SawyerPushXYMultiEnv(
                    task_id=40, 
                    hide_goal=True,
                    texture=True,
                    pos_noise=0.01,
                    randomize_gripper=False,
                    forward_only=False,
                    ),
                )
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
            if self._variant['perception'] == 'autoencoder':
                goal_images = env._env.env.get_expert_images()
                goal_aefeatures = env.feature_points(goal_images)
                goal_examples = goal_aefeatures
            elif self._variant['perception'] == 'full_state':
                goal_examples = env._env.env.get_expert_fullstates()
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

        initialize_tf_variables(self._session, only_uninitialized=True)

        self._built = True

    @property
    def picklables(self):
        return {
            'variant': self._variant,
            'env': self.env,
            'sampler': self.sampler,
            'algorithm': self.algorithm,
            'Qs': self.Qs,
            'classifier': self.reward_classifier,
            'policy_weights': self.policy.get_weights(),
        }


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
