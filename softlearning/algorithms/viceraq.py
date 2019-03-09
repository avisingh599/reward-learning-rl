import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util

from .sac import td_target
from .vice import VICE

class VICERAQ(VICE):
    def __init__(
            self,
            **kwargs,
    ):
        super(VICE, self).__init__(**kwargs)

    def _epoch_after_hook(self, *args, **kwargs):
        #TODO Avi this code is repeated from RAQ
        #figure out some clean way to reuse it

        batch_of_interest = self._pool.last_n_batch(self._epoch_length)
        observations_of_interest = batch_of_interest['observations']
        labels_of_interest = batch_of_interest['is_goal']

        rewards_of_interest = self._session.run(self._reward_t, feed_dict={
                                    self._observations_ph: observations_of_interest})

        #TODO Avi maybe log this quantity
        max_ind = np.argmax(rewards_of_interest)

        if labels_of_interest[max_ind]:
            self._goal_examples = np.concatenate([
                    self._goal_examples,
                    np.expand_dims(observations_of_interest[max_ind], axis=0) 
                    ])
        #TODO Avi Figure out if it makes sense to use these
        #"hard" negatives in some interesting way
        # else:
        #     self._negative_examples = np.concatenate([
        #             self._negative_examples,
        #             np.expand_dims(observations_of_interest[max_ind], axis=0) 
        #             ])

        for i in range(self._n_classifier_train_steps):
            feed_dict = self._get_classifier_feed_dict()
            self._train_classifier_step(feed_dict)

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        diagnostics = super(VICE, self).get_diagnostics(
            iteration, batch, training_paths, evaluation_paths)

        diagnostics.update({
            'active_learning/positives-set-size': self._goal_examples.shape[0],
            # 'active_learning/negatives-set-size': self._negative_examples.shape[0],
        })

        return diagnostics