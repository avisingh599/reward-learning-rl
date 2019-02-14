import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util

from .sac import td_target
from .sac_classifier import SACClassifier

class RAQ(SACClassifier):
    def __init__(
            self,
            **kwargs,
    ):
        super(RAQ, self).__init__(**kwargs)

        goal_example_feature_dim = self._goal_examples.shape[1]
        self._negative_examples = np.empty((0, goal_example_feature_dim))
        self._last_active_query_idx = 0

    def _get_classifier_feed_dict(self):
        rand_positive_ind = np.random.randint(
            self._goal_examples.shape[0], 
            size=self._classifier_batch_size)
        rand_negative_ind = np.random.randint(
            self._negative_examples.shape[0], 
            size=self._classifier_batch_size)

        positives = self._goal_examples[rand_positive_ind]
        negatives = self._negative_examples[rand_negative_ind]

        labels_batch = np.zeros((2*self._classifier_batch_size,1))
        labels_batch[self._classifier_batch_size:] = 1.0
        observation_batch = np.concatenate([negatives, positives], axis=0)
        
        feed_dict = {
            self._observations_ph: observation_batch, 
            self._label_ph: labels_batch,
        }

        return feed_dict

    def _add_randomly_collected_negatives(self):
        assert(self._pool._pointer == self._pool._size)
        self._negative_examples = self._pool.fields['observations'][:self._pool._pointer]

    def _epoch_after_hook(self, *args, **kwargs):
        #do the active query here
        #forward recent samples from the replay pool though the classifier
        #select ind with highest probability/reward
        #add it to set of positives or negatives based on whether it is positve or not
        observations_of_interest = self._pool.fields['observations'][
                        self._last_active_query_idx:self._pool._pointer]
        labels_of_interest = self._pool.fields['is_goal'][
                        self._last_active_query_idx:self._pool._pointer]

        self._last_active_query_idx = self._pool._pointer 
        rewards_of_interest = self._session.run(self._reward_t, feed_dict={
                                    self._observations_ph: observations_of_interest})

        max_ind = np.argmax(rewards_of_interest)

        if labels_of_interest[max_ind]:
            self._goal_examples = np.concatenate([
                    self._goal_examples,
                    np.expand_dims(observations_of_interest[max_ind], axis=0) 
                    ])
        else:
            self._negative_examples = np.concatenate([
                    self._negative_examples,
                    np.expand_dims(observations_of_interest[max_ind], axis=0) 
                    ])
        #train/re-train the classifier
        if self._epoch == 0:
            self._add_randomly_collected_negatives()
            n_train_steps = self._n_classifier_train_steps_init
        else:
            n_train_steps = self._n_classifier_train_steps_update

        for i in range(n_train_steps):
            feed_dict = self._get_classifier_feed_dict()
            self._train_classifier_step(feed_dict)
        

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        diagnostics = super(RAQ, self).get_diagnostics(
            iteration, batch, training_paths, evaluation_paths)

        diagnostics.update({
            'active_learning/positives-set-size': self._goal_examples.shape[0],
            'active_learning/negatives-set-size': self._negative_examples.shape[0],
        })

        return diagnostics
