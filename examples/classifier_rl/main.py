import os
import copy
import glob
import pickle
import sys

import tensorflow as tf
import numpy as np
from ray import tune

from softlearning.environments.utils import get_goal_example_environment_from_variant
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import get_policy_from_variant, get_policy
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant
from softlearning.models.utils import get_reward_classifier_from_variant
from softlearning.misc.generate_goal_examples import get_goal_example_from_variant

from softlearning.misc.utils import set_seed, initialize_tf_variables
from examples.instrument import run_example_local
from examples.development.main import ExperimentRunner

class ExperimentRunnerClassifierRL(ExperimentRunner):

    def _build(self):
        variant = copy.deepcopy(self._variant)

        env = self.env = get_goal_example_environment_from_variant(variant)
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

            goal_examples_train, goal_examples_validation = \
                get_goal_example_from_variant(variant)
            algorithm_kwargs['goal_examples'] = goal_examples_train
            algorithm_kwargs['goal_examples_validation'] = \
                goal_examples_validation

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

            goal_examples_train, goal_examples_validation = \
                get_goal_example_from_variant(variant)
            algorithm_kwargs['goal_examples'] = goal_examples_train
            algorithm_kwargs['goal_examples_validation'] = \
                goal_examples_validation

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
