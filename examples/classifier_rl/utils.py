import multiprocessing
import argparse
from distutils.util import strtobool
import json
import os

import softlearning.algorithms.utils as alg_utils
import softlearning.environments.utils as env_utils
from softlearning.misc.utils import datetimestamp

DEFAULT_TASK = 'StateDoorPullEnv-v0'
DEFAULT_ALGORITHM = 'VICE'
AVAILABLE_ALGORITHMS = set(alg_utils.ALGORITHM_CLASSES.keys())

import gym
from multiworld.envs.mujoco import register_goal_example_envs
envs_before = set(env_spec.id for env_spec in gym.envs.registry.all())
register_goal_example_envs()
envs_after = set(env_spec.id for env_spec in gym.envs.registry.all())
goal_example_envs = tuple(sorted(envs_after - envs_before))


def add_ray_init_args(parser):

    def init_help_string(help_string):
        return help_string + " Passed to `ray.init`."

    parser.add_argument(
        '--cpus',
        type=int,
        default=None,
        help=init_help_string("Cpus to allocate to ray process."))
    parser.add_argument(
        '--gpus',
        type=int,
        default=None,
        help=init_help_string("Gpus to allocate to ray process."))
    parser.add_argument(
        '--resources',
        type=json.loads,
        default=None,
        help=init_help_string("Resources to allocate to ray process."))
    parser.add_argument(
        '--include-webui',
        type=str,
        default=True,
        help=init_help_string("Boolean flag indicating whether to start the"
                              "web UI, which is a Jupyter notebook."))
    parser.add_argument(
        '--temp-dir',
        type=str,
        default=None,
        help=init_help_string("If provided, it will specify the root temporary"
                              " directory for the Ray process."))

    return parser


def add_ray_tune_args(parser):

    def tune_help_string(help_string):
        return help_string + " Passed to `tune.run_experiments`."

    parser.add_argument(
        '--resources-per-trial',
        type=json.loads,
        default={},
        help=tune_help_string("Resources to allocate for each trial."))
    parser.add_argument(
        '--trial-gpus',
        type=float,
        default=None,
        help=("Resources to allocate for each trial. Passed"
              " to `tune.run_experiments`."))
    parser.add_argument(
        '--trial-extra-cpus',
        type=int,
        default=None,
        help=("Extra CPUs to reserve in case the trials need to"
              " launch additional Ray actors that use CPUs."))
    parser.add_argument(
        '--trial-extra-gpus',
        type=float,
        default=None,
        help=("Extra GPUs to reserve in case the trials need to"
              " launch additional Ray actors that use GPUs."))
    parser.add_argument(
        '--num-samples',
        default=1,
        type=int,
        help=tune_help_string("Number of times to repeat each trial."))
    parser.add_argument(
        '--upload-dir',
        type=str,
        default='',
        help=tune_help_string("Optional URI to sync training results to (e.g."
                              " s3://<bucket> or gs://<bucket>)."))
    parser.add_argument(
        '--trial-name-creator',
        default=None,
        help=tune_help_string(
            "Optional creator function for the trial string, used in "
            "generating a trial directory."))
    parser.add_argument(
        '--trial-name-template',
        type=str,
        default='{trial.trial_id}-algorithm={trial.config[algorithm_params]'
                '[type]}-seed={trial.config[run_params][seed]}',
        help=tune_help_string(
            "Optional string template for trial name. For example:"
            " '{trial.trial_id}-seed={trial.config[run_params][seed]}'"))
    parser.add_argument(
        '--trial-cpus',
        type=int,
        default=multiprocessing.cpu_count(),
        help=tune_help_string("Resources to allocate for each trial."))
    parser.add_argument(
        '--checkpoint-frequency',
        type=int,
        default=0,
        help=tune_help_string(
            "How many training iterations between checkpoints."
            " A value of 0 (default) disables checkpointing. If set,"
            " takes precedence over variant['run_params']"
            "['checkpoint_frequency']."))
    parser.add_argument(
        '--checkpoint-at-end',
        type=lambda x: bool(strtobool(x)),
        default=None,
        help=tune_help_string(
            "Whether to checkpoint at the end of the experiment. If set,"
            " takes precedence over variant['run_params']"
            "['checkpoint_at_end']."))
    parser.add_argument(
        '--max-failures',
        default=0,
        type=int,
        help=tune_help_string(
            "Try to recover a trial from its last checkpoint at least this "
            "many times. Only applies if checkpointing is enabled."))
    parser.add_argument(
        '--restore',
        type=str,
        default=None,
        help=tune_help_string(
            "Path to checkpoint. Only makes sense to set if running 1 trial."
            " Defaults to None."))
    parser.add_argument(
        '--with-server',
        type=str,
        default=False,
        help=tune_help_string("Starts a background Tune server. Needed for"
                              " using the Client API."))

    return parser


def get_parser(allow_policy_list=False):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--universe', type=str, default='multiworld', choices=('multiworld', ))
    parser.add_argument(
        '--domain', type=str, default='mujoco', choices=('mujoco',))
    parser.add_argument(
        '--task', type=str, default=DEFAULT_TASK, 
        choices=goal_example_envs)
    parser.add_argument(
        '--n_goal_examples', type=int, default=10)
    parser.add_argument(
        '--n_epochs', type=int, default=200)
    parser.add_argument(
        '--active_query_frequency', type=int, default=5)

    parser.add_argument(
        '--checkpoint-replay-pool',
        type=lambda x: bool(strtobool(x)),
        default=None,
        help=("Whether a checkpoint should also saved the replay"
              " pool. If set, takes precedence over"
              " variant['run_params']['checkpoint_replay_pool']."
              " Note that the replay pool is saved (and "
              " constructed) piece by piece so that each"
              " experience is saved only once."))

    parser.add_argument(
        '--algorithm',
        type=str,
        choices=AVAILABLE_ALGORITHMS,
        default=DEFAULT_ALGORITHM)
    if allow_policy_list:
        parser.add_argument(
            '--policy',
            type=str,
            nargs='+',
            choices=('gaussian', ),
            default='gaussian')
    else:
        parser.add_argument(
            '--policy',
            type=str,
            choices=('gaussian', ),
            default='gaussian')

    parser.add_argument(
        '--exp-name',
        type=str,
        default=datetimestamp())
    parser.add_argument(
        '--mode', type=str, default='local')
    parser.add_argument(
        '--confirm-remote',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=True,
        help="Whether or not to query yes/no on remote run.")

    parser.add_argument(
        '--video-save-frequency',
        type=int,
        default=0,
        help="Save frequency for videos.")
    parser.add_argument(
        '--path-save-frequency',
        type=int,
        default=0,
        help="Save frequency for paths.")

    parser = add_ray_init_args(parser)
    parser = add_ray_tune_args(parser)
    # parser = add_ray_autoscaler_exec_args(parser)

    return parser


def variant_equals(*keys):
    def get_from_spec(spec):
        # TODO(hartikainen): This may break in some cases. ray.tune seems to
        # add a 'config' key at the top of the spec, whereas `generate_variants`
        # does not.
        node = spec.get('config', spec)
        for key in keys:
            node = node[key]

        return node

    return get_from_spec
