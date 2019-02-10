from collections import OrderedDict
from numbers import Number

import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util

from .sac import SAC

class SACClassifier(SAC):
    def __init__(
            self,
            classifier,
            classifier_train_steps=int(1e4),
            classifier_optim='sgd',
            **kwargs,
    ):

        super(SACClassifier, self).__init__(**kwargs)