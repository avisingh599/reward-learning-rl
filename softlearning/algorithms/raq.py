import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util

from .sac import td_target
from .sac_classifier import SACClassifier

class RAQ(SACClassifier):
    def __init__(
            self,
            active_query_frequency=5,
            **kwargs,
    ):
        super(RAQ, self).__init__(**kwargs)

        goal_example_feature_dim = self._goal_examples.shape[1]
        self._negative_examples = np.empty((0, goal_example_feature_dim))
        self._active_query_frequency = active_query_frequency

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

        if self._epoch % self._active_query_frequency == 0:

            batch_of_interest = self._pool.last_n_batch(self._epoch_length*self._active_query_frequency)
            observations_of_interest = batch_of_interest['observations']
            labels_of_interest = batch_of_interest['is_goal']

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

        for i in range(self._n_classifier_train_steps):
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
