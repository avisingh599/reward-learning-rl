"""Implements a GymAdapter that converts Gym envs into SoftlearningEnv."""

import numpy as np
import gym
from gym import spaces, wrappers

import torch
from torch.autograd import Variable

from multiworld.envs.pygame.point2d import Point2DEnv, Point2DWallEnv

from .softlearning_env import SoftlearningEnv
from softlearning.environments.gym.wrappers import NormalizeActionWrapper
from softlearning.environments.gym.mujoco.ant_env import AntEnv as CustomAntEnv
from softlearning.environments.gym.mujoco.humanoid_env import (
    HumanoidEnv as CustomHumanoidEnv)
from softlearning.environments.gym.mujoco.walker2d_env import (
    Walker2dEnv as CustomWalker2dEnv)
from softlearning.environments.gym.mujoco.half_cheetah_env import (
    HalfCheetahEnv as CustomHalfCheetahEnv)
from softlearning.environments.gym.mujoco.hopper_env import (
    HopperEnv as CustomHopperEnv)
from softlearning.environments.gym.mujoco.swimmer_env import (
    SwimmerEnv as CustomSwimmerEnv)
from softlearning.environments.gym.mujoco.pusher_2d_env import (
    Pusher2dEnv,
    ForkReacherEnv)
from softlearning.environments.gym.mujoco.image_pusher import (
    ImagePusherEnv,
    ImageForkReacherEnv,
    BlindForkReacherEnv)
from softlearning.environments.gym.multi_goal import MultiGoalEnv


def raise_on_use_wrapper(e):
    def raise_on_use(*args, **kwargs):
        raise e
    return raise_on_use


GYM_ENVIRONMENTS = {
    'Swimmer': {
        'v2': lambda: gym.envs.make('Swimmer-v2'),
        'Custom': CustomSwimmerEnv,
        'Default': lambda: gym.envs.make('Swimmer-v2'),
    },
    'Ant': {
        'v2': lambda: gym.envs.make('Ant-v2'),
        'Custom': CustomAntEnv,
        'Default': lambda: gym.envs.make('Ant-v2'),
    },
    'Humanoid': {
        'v2': lambda: gym.envs.make('Humanoid-v2'),
        'Standup-v2': lambda: gym.envs.make('HumanoidStandup-v2'),
        'Custom': CustomHumanoidEnv,
        'Default': lambda: gym.envs.make('Humanoid-v2'),
    },
    'Hopper': {
        'v2': lambda: gym.envs.make('Hopper-v2'),
        'Custom': CustomHopperEnv,
        'Default': lambda: gym.envs.make('Hopper-v2'),
    },
    'HalfCheetah': {
        'v2': lambda: gym.envs.make('HalfCheetah-v2'),
        'Custom': CustomHalfCheetahEnv,
        'Default': lambda: gym.envs.make('HalfCheetah-v2'),
    },
    'Walker': {
        'v2': lambda: gym.envs.make('Walker2d-v2'),
        'Custom': CustomWalker2dEnv,
        'Default': lambda: gym.envs.make('Walker2d-v2'),
    },
    'Pusher2d': {
        'Default': Pusher2dEnv,
        'DefaultReach': ForkReacherEnv,

        'ImageDefault': ImagePusherEnv,
        'ImageReach': ImageForkReacherEnv,
        'BlindReach': BlindForkReacherEnv,
    },
    'Point2DEnv': {
        'Default': Point2DEnv,
        'Wall': Point2DWallEnv,
    },
    'HandManipulatePen': {
        'v0': lambda: gym.envs.make('HandManipulatePen-v0'),
        'Dense-v0': lambda: gym.envs.make('HandManipulatePenDense-v0'),
        'Default': lambda: gym.envs.make('HandManipulatePen-v0'),
    },
    'HandManipulateEgg': {
        'v0': lambda: gym.envs.make('HandManipulateEgg-v0'),
        'Dense-v0': lambda: gym.envs.make('HandManipulateEggDense-v0'),
        'Default': lambda: gym.envs.make('HandManipulateEgg-v0'),
    },
    'HandManipulateBlock': {
        'v0': lambda: gym.envs.make('HandManipulateBlock-v0'),
        'Dense-v0': lambda: gym.envs.make('HandManipulateBlockDense-v0'),
        'Default': lambda: gym.envs.make('HandManipulateBlock-v0'),
    },
    'HandReach': {
        'v0': lambda: gym.envs.make('HandReach-v0'),
        'Dense-v0': lambda: gym.envs.make('HandReachDense-v0'),
        'Default': lambda: gym.envs.make('HandReach-v0'),
    },
    'InvertedDoublePendulum': {
        'Default': lambda: gym.envs.make('InvertedDoublePendulum-v2'),
        'v2': lambda: gym.envs.make('InvertedDoublePendulum-v2'),
    },
    'Reacher': {
        'Default': lambda: gym.envs.make('Reacher-v2'),
        'v2': lambda: gym.envs.make('Reacher-v2'),
    },
    'InvertedPendulum': {
        'Default': lambda: gym.envs.make('InvertedPendulum-v2'),
        'v2': lambda: gym.envs.make('InvertedPendulum-v2'),
    },
    'MultiGoal': {
        'Default': MultiGoalEnv
    },
}


class GymAdapter(SoftlearningEnv):
    """Adapter that implements the SoftlearningEnv for Gym envs."""

    def __init__(self,
                 *args,
                 domain=None,
                 task=None,
                 env=None,
                 normalize=True,
                 observation_keys=None,
                 unwrap_time_limit=True,
                 **kwargs):
        self.normalize = normalize
        self.observation_keys = observation_keys
        self.unwrap_time_limit = unwrap_time_limit

        self._Serializable__initialize(locals())
        super(GymAdapter, self).__init__(domain, task, *args, **kwargs)

        if env is not None:
            assert domain is None and task is None
            env = env
        else:
            assert domain is not None and task is not None
            env = GYM_ENVIRONMENTS[domain][task](*args, **kwargs)

        if isinstance(env, wrappers.TimeLimit) and unwrap_time_limit:
            # Remove the TimeLimit wrapper that sets 'done = True' when
            # the time limit specified for each environment has been passed and
            # therefore the environment is not Markovian (terminal condition
            # depends on time rather than state).
            env = env.env

        if isinstance(env.observation_space, spaces.Dict):
            observation_keys = (
                observation_keys or list(env.observation_space.spaces.keys()))
        if normalize:
            env = NormalizeActionWrapper(env)

        self._env = env

    @property
    def observation_space(self):
        observation_space = self._env.observation_space
        return observation_space

    @property
    def active_observation_shape(self):
        """Shape for the active observation based on observation_keys."""
        if not isinstance(self._env.observation_space, spaces.Dict):
            return super(GymAdapter, self).active_observation_shape

        observation_keys = (
            self.observation_keys
            or list(self._env.observation_space.spaces.keys()))

        active_size = sum(
            np.prod(self._env.observation_space.spaces[key].shape)
            for key in observation_keys)

        active_observation_shape = (active_size, )

        return active_observation_shape

    def convert_to_active_observation(self, observation):
        if not isinstance(self._env.observation_space, spaces.Dict):
            return observation

        observation_keys = (
            self.observation_keys
            or list(self._env.observation_space.spaces.keys()))

        observation = np.concatenate([
            observation[key] for key in observation_keys
        ], axis=-1)

        return observation

    @property
    def action_space(self, *args, **kwargs):
        action_space = self._env.action_space
        if len(action_space.shape) > 1:
            raise NotImplementedError(
                "Action space ({}) is not flat, make sure to check the"
                " implemenation.".format(action_space))
        return action_space

    def step(self, action, *args, **kwargs):
        # TODO(hartikainen): refactor this to always return an OrderedDict,
        # such that the observations for all the envs is consistent. Right now
        # some of the gym envs return np.array whereas others return dict.
        #
        # Something like:
        # observation = OrderedDict()
        # observation['observation'] = env.step(action, *args, **kwargs)
        # return observation

        return self._env.step(action, *args, **kwargs)

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self._env.close(*args, **kwargs)

    def seed(self, *args, **kwargs):
        return self._env.seed(*args, **kwargs)

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def get_param_values(self, *args, **kwargs):
        raise NotImplementedError

    def set_param_values(self, *args, **kwargs):
        raise NotImplementedError

class GymAdapterAutoEncoder(GymAdapter):

    def __init__(self,
                 autoencoder_model,
                 autoencoder_savepath,
                 *args,
                 domain=None,
                 task=None,
                 env=None,
                 use_jointstate=False,
                 normalize=True,
                 observation_keys=None,
                 unwrap_time_limit=True,
                 **kwargs):
        #self.normalize = normalize
        #self.observation_keys = observation_keys
        #self.unwrap_time_limit = unwrap_time_limit

        self._Serializable__initialize(locals())
        super(GymAdapterAutoEncoder, self).__init__(
                                    *args,
                                    domain=domain, task=task, env=env, 
                                    normalize=normalize, 
                                    observation_keys=observation_keys,
                                    unwrap_time_limit=unwrap_time_limit, 
                                    **kwargs)
        self._autoencoder_savepath = autoencoder_savepath
        self._use_jointstate = use_jointstate
        if use_jointstate:
            self._joint_space = self._env.env.joint_space

        # import pdb; pdb.set_trace()
        # if isinstance(autoencoder_savepath, str):
        #     self._autoencoder = torch.load(autoencoder_savepath)
        # else:
        #     self._autoencoder = autoencoder_savepath

        #import pdb; pdb.set_trace()
        self._autoencoder = autoencoder_model
        self._autoencoder.load_state_dict(torch.load(autoencoder_savepath))
        self._autoencoder.cuda()
        self._autoencoder.eval()

        #self._autoencoder.eval()
        #import pdb; pdb.set_trace()
        image = np.zeros(
            (self._env.env.camera_height, self._env.env.camera_width, self._env.env.camera_channels),
            dtype=np.float32
        )
        feature_points = self.feature_points(image)
        self._feature_dim = feature_points.shape[-1]

        if self._use_jointstate:
            self._joint_dim = self._env.env.joint_space.flat_dim
            new_dim = self._joint_dim + self._feature_dim
        else:
            new_dim = self._feature_dim
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=[new_dim],
        )


    def step(self, action, *args, **kwargs):
        _obs, reward, done, env_infos = self._env.step(action, *args, **kwargs)
        obs = self._get_obs()
        return obs, reward, done, env_infos

    # def step(self, action, *args, **kwargs):
    #     # TODO(hartikainen): refactor this to always return an OrderedDict,
    #     # such that the observations for all the envs is consistent. Right now
    #     # some of the gym envs return np.array whereas others return dict.
    #     #
    #     # Something like:
    #     # observation = OrderedDict()
    #     # observation['observation'] = env.step(action, *args, **kwargs)
    #     # return observation

    #     return self._env.step(action, *args, **kwargs)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def joint_space(self):
        return self._joint_space

    def feature_points(self, image):
        """Image in HWC format"""
        dim = len(image.shape)
        if dim == 3:
            image = image[None]
        image = image.transpose((0, 3, 1, 2))
        torch_image = torch.from_numpy(image)
        torch_image = Variable(torch_image.cuda()).float()
        torch_fp = self._autoencoder.features(torch_image)
        fp = torch_fp.data.cpu().numpy()
        if dim == 3:
            fp = fp[0]
        return fp

    def reconstruction(self, image):
        dim = len(image.shape)
        if dim == 3:
            image = image[None]
        image = image.transpose((0, 3, 1, 2))
        torch_image = torch.from_numpy(image)
        torch_image = Variable(torch_image.cuda()).float()
        reconstruction = self._autoencoder.reconstruction(torch_image)
        reconstruction = reconstruction.data.cpu().numpy()
        if dim == 3:
            reconstruction = reconstruction[0]
        reconstruction = reconstruction.transpose((1, 2, 0))
        return reconstruction

    def reset(self):
        self._env.reset()
        return self._get_obs()

    def _get_obs(self):
        image = self._env.env.get_image()
        fp = self.feature_points(image)

        if self._use_jointstate:
            joint_obs = self._env.env._get_jointstate()
            return np.concatenate([fp, joint_obs])
        else:
            return fp