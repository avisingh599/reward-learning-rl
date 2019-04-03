import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util

from .sac import td_target
from .sac_classifier import SACClassifier

class VICE(SACClassifier):
    """Varitational Inverse Control with Events (VICE)

    References
    ----------
    [1] Variational Inverse Control with Events: A General
    Framework for Data-Driven Reward Definition. Justin Fu, Avi Singh,
    Dibya Ghosh, Larry Yang, Sergey Levine, NIPS 2018.
    """
    def _get_Q_target(self):
        next_actions = self._policy.actions([self._next_observations_ph])
        next_log_pis = self._policy.log_pis(
            [self._next_observations_ph], next_actions)

        next_Qs_values = tuple(
            Q([self._next_observations_ph, next_actions])
            for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_value = min_next_Q - self._alpha * next_log_pis

        observation_logits = self._classifier([self._observations_ph])
        self._reward_t = observation_logits

        Q_target = td_target(
            reward=self._reward_scale * self._reward_t,
            discount=self._discount,
            next_value=(1 - self._terminals_ph) * next_value)

        return Q_target

    def _init_classifier_update(self):
        log_p = self._classifier([self._observations_ph])
        sampled_actions = self._policy.actions([self._observations_ph])
        log_pi = self._policy.log_pis([self._observations_ph], sampled_actions)
        log_pi_log_p_concat = tf.concat(
            [tf.expand_dims(log_pi, 1), tf.expand_dims(log_p, 1)], axis=1)
        label_onehot = tf.one_hot(tf.cast(self._label_ph, tf.int32), 2, dtype=tf.int32)
        self._classifier_loss_t = tf.reduce_mean(
            tf.losses.softmax_cross_entropy(
                label_onehot,
                log_pi_log_p_concat,
            )
        )
        self._classifier_training_op = self._get_classifier_training_op()

    def _epoch_after_hook(self, *args, **kwargs):
        for i in range(self._n_classifier_train_steps):
            feed_dict = self._get_classifier_feed_dict()
            self._train_classifier_step(feed_dict)
