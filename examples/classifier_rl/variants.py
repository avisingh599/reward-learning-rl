from ray import tune
import numpy as np

from softlearning.misc.utils import get_git_rev, deep_update
from softlearning.misc.generate_goal_examples import DOOR_TASKS, PUSH_TASKS

M = 256
REPARAMETERIZE = True

NUM_COUPLING_LAYERS = 2

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True,
    }
}

GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN = {}

POLICY_PARAMS_BASE = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_BASE,
}

POLICY_PARAMS_BASE.update({
    'gaussian': POLICY_PARAMS_BASE['GaussianPolicy'],
})

POLICY_PARAMS_FOR_DOMAIN = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN,
}

POLICY_PARAMS_FOR_DOMAIN.update({
    'gaussian': POLICY_PARAMS_FOR_DOMAIN['GaussianPolicy'],
})

DEFAULT_MAX_PATH_LENGTH = 100
MAX_PATH_LENGTH_PER_DOMAIN = {
    'Point2DEnv': 50,
}

ALGORITHM_PARAMS_BASE = {
    'type': 'SAC',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_render_mode': None,
        'eval_n_episodes': 5,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,
    }
}

#TODO Avi Most of the algorithm params for classifier-style methods
#are shared. Rewrite this part to reuse the params
ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'type': 'SAC',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(1e3),
            'n_epochs': 200,
        }
    },
    'SACClassifier': {
        'type': 'SACClassifier',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'classifier_lr': 1e-4,
            'classifier_batch_size': 128,
            'n_initial_exploration_steps': int(1e3),
            'n_classifier_train_steps': 10000,
            #'n_classifier_train_steps': tune.grid_search([10, 100, 1000]),
            # 'classifier_optim_name': tune.grid_search(['adam', 'sgd']),
            'classifier_optim_name': 'adam',
            'reward_type': 'logits',
            'n_epochs': 200,
        }
    },
    'RAQ': {
        'type': 'RAQ',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            #'target_entropy': tune.grid_search([-10, -7, -5, -2, 0, 5]),
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'classifier_lr': 1e-4,
            'classifier_batch_size': 128,
            'n_initial_exploration_steps': int(1e3),
            'n_classifier_train_steps': 100,
            #'n_classifier_train_steps': tune.grid_search([10, 100, 1000]),
            # 'classifier_optim_name': tune.grid_search(['adam', 'sgd']),
            'classifier_optim_name': 'adam',
            'reward_type': 'logits',
            'n_epochs': 200,
        }
    },
    'VICE': {
        'type': 'VICE',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'classifier_lr': 1e-4,
            'classifier_batch_size': 128,
            'n_initial_exploration_steps': int(1e3),
            'n_classifier_train_steps': 100,
            #'n_classifier_train_steps': tune.grid_search([10, 100, 1000]),
            # 'classifier_optim_name': tune.grid_search(['adam', 'sgd']),
            'classifier_optim_name': 'adam',
            'n_epochs': 200,
        }
    },
    'VICERAQ': {
        'type': 'VICERAQ',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'classifier_lr': 1e-4,
            'classifier_batch_size': 128,
            'n_initial_exploration_steps': int(1e3),
            'n_classifier_train_steps': 100,
            # 'n_classifier_train_steps': tune.grid_search([10, 100, 1000]),
            # 'classifier_optim_name': tune.grid_search(['adam', 'sgd']),
            'classifier_optim_name': 'adam',
            'n_epochs': 200,
        }
    },
    'SQL': {
        'type': 'SQL',
        'kwargs': {
            'policy_lr': 3e-4,
            'td_target_update_interval': 1,
            'n_initial_exploration_steps': int(1e3),
            'reward_scale': tune.sample_from(lambda spec: (
                {
                    'Swimmer': 30,
                    'Hopper': 30,
                    'HalfCheetah': 30,
                    'Walker': 10,
                    'Ant': 300,
                    'Humanoid': 100,
                }[spec.get('config', spec)['domain']],
            ))
        }
    }
}

DEFAULT_NUM_EPOCHS = 200
NUM_CHECKPOINTS = 10


def get_variant_spec_base(universe, domain, task, policy, algorithm):
    # algorithm_params = deep_update(
    #     ALGORITHM_PARAMS_BASE,
    #     ALGORITHM_PARAMS_PER_DOMAIN.get(domain, {})
    # )
    # algorithm_params = deep_update(
    #     algorithm_params,
    #     ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
    # )
    algorithm_params = ALGORITHM_PARAMS_BASE
    algorithm_params = deep_update(
            algorithm_params,
            ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
        )

    variant_spec = {
        'domain': domain,
        'task': task,
        'universe': universe,
        'git_sha': get_git_rev(),

        #'env_params': ENV_PARAMS.get(domain, {}).get(task, {}),
        'policy_params': deep_update(
            POLICY_PARAMS_BASE[policy],
            POLICY_PARAMS_FOR_DOMAIN[policy].get(domain, {})
        ),
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, M),
            }
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': 1e6,
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'min_pool_size': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'batch_size': 256,
            }
        },
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency':  DEFAULT_NUM_EPOCHS // NUM_CHECKPOINTS,
            'checkpoint_replay_pool': False,
        },
    }

    return variant_spec


def get_variant_spec_classifier(universe,
                                domain,
                                task,
                                policy,
                                algorithm,
                                n_goal_examples,
                                *args,
                                **kwargs):
    variant_spec = get_variant_spec_base(
        universe, domain, task, policy, algorithm, *args, **kwargs)

    classifier_layer_size = L = 256
    variant_spec['classifier_params'] = {
        'type': 'feedforward_classifier',
        'kwargs': {
            'hidden_layer_sizes': (L,L),
            }
        }

    variant_spec['data_params'] = {
        'n_goal_examples': n_goal_examples,
        'n_goal_examples_validation_max': 100,
    }

    if algorithm in ['RAQ', 'VICERAQ']:

        if task in DOOR_TASKS:
            is_goal_key = 'angle_success'
        elif task in PUSH_TASKS:
            is_goal_key = 'puck_success'
        else:
            raise NotImplementedError('Success metric not defined for task')

        variant_spec.update({

            'sampler_params': {
                'type': 'ActiveSampler',
                'kwargs': {
                    'is_goal_key': is_goal_key,
                    'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                        domain, DEFAULT_MAX_PATH_LENGTH),
                    'min_pool_size': MAX_PATH_LENGTH_PER_DOMAIN.get(
                        domain, DEFAULT_MAX_PATH_LENGTH),
                    'batch_size': 256,
                }
            },
            'replay_pool_params': {
                'type': 'ActiveReplayPool',
                'kwargs': {
                    'max_size': 1e6,
                }
            },

            #TODO Avi implement variable active query frequency
            'active_params': {
                'active_query_frequency': 1,
            }

            })

    return variant_spec

def get_variant_spec(args):
    universe, domain = args.universe, args.domain
    task, algorithm = args.task, args.algorithm

    if args.algorithm in ['SACClassifier', 'RAQ', 'VICE', 'VICERAQ']:
        variant_spec = get_variant_spec_classifier(
            universe, domain, task, args.policy, args.algorithm,
            args.n_goal_examples)
    else:
        variant_spec = get_variant_spec_base(
            universe, domain, task, args.policy, args.algorithm)

    if 'Image48' in task:
        preprocessor_params = {
            'type': 'convnet_preprocessor',
            'kwargs': {
                #'image_shape': variant_spec['env_params']['image_shape'],
                'image_shape': (48, 48, 3),
                'output_size': M,
                'conv_filters': (8, 8),
                'conv_kernel_sizes': ((5, 5), (5, 5)),
                'pool_type': 'MaxPool2D',
                'pool_sizes': ((2, 2), (2, 2)),
                'pool_strides': (2, 2),
                'dense_hidden_layer_sizes': (),
            },
        }
        variant_spec['policy_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())
        variant_spec['Q_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())
        variant_spec['replay_pool_params']['kwargs']['max_size'] = 3e5

        if args.algorithm in ['SACClassifier', 'RAQ', 'VICE', 'VICERAQ']:
            variant_spec['classifier_params']['kwargs']['preprocessor_params'] = (
                preprocessor_params.copy())

    elif 'Image' in task:
        raise NotImplementedError('Add convnet preprocessor for this image input')

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
