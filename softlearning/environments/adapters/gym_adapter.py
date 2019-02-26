"""Implements a GymAdapter that converts Gym envs into SoftlearningEnv."""

import numpy as np
import gym
from gym import spaces, wrappers

import torch
from torch.autograd import Variable

from .softlearning_env import SoftlearningEnv
from softlearning.environments.gym import register_environments
from softlearning.environments.gym.wrappers import NormalizeActionWrapper
from collections import defaultdict

import tensorflow as tf
from softlearning.utils.keras import PicklableKerasModel

def parse_domain_task(gym_id):
    domain_task_parts = gym_id.split('-')
    domain = '-'.join(domain_task_parts[:1])
    task = '-'.join(domain_task_parts[1:])

    return domain, task


CUSTOM_GYM_ENVIRONMENT_IDS = register_environments()
CUSTOM_GYM_ENVIRONMENTS = defaultdict(list)

for gym_id in CUSTOM_GYM_ENVIRONMENT_IDS:
    domain, task = parse_domain_task(gym_id)
    CUSTOM_GYM_ENVIRONMENTS[domain].append(task)

CUSTOM_GYM_ENVIRONMENTS = dict(CUSTOM_GYM_ENVIRONMENTS)

GYM_ENVIRONMENT_IDS = tuple(gym.envs.registry.env_specs.keys())
GYM_ENVIRONMENTS = defaultdict(list)


for gym_id in GYM_ENVIRONMENT_IDS:
    domain, task = parse_domain_task(gym_id)
    GYM_ENVIRONMENTS[domain].append(task)

GYM_ENVIRONMENTS = dict(GYM_ENVIRONMENTS)


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
        assert not args, (
            "Gym environments don't support args. Use kwargs instead.")

        self.normalize = normalize
        self.observation_keys = observation_keys
        self.unwrap_time_limit = unwrap_time_limit

        self._Serializable__initialize(locals())
        super(GymAdapter, self).__init__(domain, task, *args, **kwargs)

        if env is None:
            assert (domain is not None and task is not None), (domain, task)
            env_id = f"{domain}-{task}"
            env = gym.envs.make(env_id, **kwargs)
        else:
            assert domain is None and task is None, (domain, task)

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

class GymAdapterPixel(GymAdapter):

    def __init__(self, *args, **kwargs):
        self._Serializable__initialize(locals())
        super(GymAdapterPixel, self).__init__(*args, **kwargs)
        img_dims = (self._env.env.camera_height, self._env.env.camera_width, self._env.env.camera_channels)

        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=[np.prod(img_dims)],
        )

    def reset(self):
        self._env.reset()
        return self._get_obs()

    def step(self, action, *args, **kwargs):
        _obs, reward, done, env_infos = self._env.step(action, *args, **kwargs)
        obs = self._get_obs()
        return obs, reward, done, env_infos

    def _get_obs(self):
        image = self._env.env.get_image()
        return image.flatten()

    @property
    def observation_space(self):
        return self._observation_space

class GymAdapterAutoEncoderTF(GymAdapter):
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
                 **kwargs
                 ):
        #super(GymAdapterAutoEncoderTF, self).__init__(**kwargs)
        self._Serializable__initialize(locals())
        super(GymAdapterAutoEncoderTF, self).__init__(
                                    *args,
                                    domain=domain, task=task, env=env,
                                    normalize=normalize,
                                    observation_keys=observation_keys,
                                    unwrap_time_limit=unwrap_time_limit,
                                    **kwargs)

        self._autoencoder_full_model = autoencoder_model
        self._use_jointstate = use_jointstate
        self._autoencoder_full_model.load_weights(autoencoder_savepath)
        self._autoencoder = PicklableKerasModel(
            inputs=self._autoencoder_full_model.inputs,
            outputs=self._autoencoder_full_model.outputs[0])

        self._feature_dim = self._autoencoder.outputs[0].shape[1].value

        if self._use_jointstate:
            self._joint_dim = self._env.env.joing_space.flat_dim
            new_dim = self._joint_dim + self._feature_dim
        else:
            new_dim = self._feature_dim

        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=[new_dim],
            )

    def feature_points(self, image):
        if len(image.shape) == 4:
            return self._autoencoder.predict(image)
        elif len(image.shape) == 3:
            return self._autoencoder.predict(
                np.expand_dims(image, axis=0))[0]
        else:
            raise NotImplementedError

    def step(self, action, *args, **kwargs):
        _obs, reward, done, env_infos = self._env.step(action, *args, **kwargs)
        obs = self._get_obs()
        return obs, reward, done, env_infos

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

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def joint_space(self):
        return self._joint_space

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
