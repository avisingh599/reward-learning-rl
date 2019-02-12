from collections import OrderedDict
from numbers import Number

import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util

from .sac import SAC, td_target

class SACClassifier(SAC):
    def __init__(
            self,
            classifier,
            goal_examples,
            classifier_lr=1e-4,
            classifier_batch_size=128,
            n_classifier_train_steps_init=int(1e4),
            n_classifier_train_steps_update=int(1e3),
            classifier_optim_name='adam',
            **kwargs,
    ):
        
        self._classifier = classifier
        self._goal_examples = goal_examples
        self._classifier_lr = classifier_lr
        self._n_classifier_train_steps_init = n_classifier_train_steps_init
        self._n_classifier_train_steps_update = n_classifier_train_steps_update
        self._classifier_optim_name = classifier_optim_name
        self._classifier_batch_size = classifier_batch_size


        super(SACClassifier, self).__init__(**kwargs)
    
    def _build(self):
        super(SACClassifier, self)._build()
        self._init_classifier_update()
    
    def _init_placeholders(self):
        super(SACClassifier, self)._init_placeholders()
        self._label_ph = tf.placeholder(
            tf.float32,
            shape=[None, 1],
            name='labels',
        )

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
        self._reward_t = tf.nn.sigmoid(observation_logits)

        Q_target = td_target(
            reward=self._reward_scale * self._reward_t,
            discount=self._discount,
            next_value=(1 - self._terminals_ph) * next_value)

        return Q_target

    def _init_classifier_update(self):
        #TODO Avi modify this to match whatever we do in VICE
        logits = self._classifier([self._observations_ph])
        #logits = tf.expand_dims(logits, axis=-1)
        # self._discriminator_t = tf.squeeze(tf.nn.sigmoid(logits), axis=-1)
        #self._discriminator_t = tf.nn.sigmoid(logits)

        #classifier_logprob = -1*tf.nn.relu(energy)
        #classifier_logprob = -10*tf.nn.sigmoid(classifier_output_linear)
        
        cross_entropy_t = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=self._label_ph)
        self._classifier_loss_t = tf.reduce_mean(cross_entropy_t)
        #self._reward_t = self._discriminator_t

        if self._classifier_optim_name == 'adam':
            self._classifier_optimizer = tf.train.AdamOptimizer(
                                        learning_rate=self._classifier_lr,
                                        name='classifier_optimizer')
        else:
            raise NotImplementedError

        self._classifier_training_op = \
            tf.contrib.layers.optimize_loss(
                self._classifier_loss_t,
                self.global_step,
                learning_rate=self._classifier_lr,
                optimizer=self._classifier_optimizer,
                variables=self._classifier.trainable_variables,
                increment_global_step=False,
                summaries=((
                    "loss", "gradients", "gradient_norm", "global_gradient_norm"
                ) if self._tf_summaries else ())
                )

    def _get_classifier_feed_dict(self):

        negatives = self.sampler.random_batch(self._classifier_batch_size)['observations']
        rand_positive_ind = np.random.randint(self._goal_examples.shape[0], size=self._classifier_batch_size)
        positives = self._goal_examples[rand_positive_ind]

        labels_batch = np.zeros((2*self._classifier_batch_size,1))
        labels_batch[self._classifier_batch_size:] = 1.0
        observation_batch = np.concatenate([negatives, positives], axis=0)
        
        feed_dict = {
            self._observations_ph: observation_batch, 
            self._label_ph: labels_batch
        }

        return feed_dict

    def _train_classifier_step(self):
        feed_dict = self._get_classifier_feed_dict()
        _, loss = self._session.run([self._classifier_training_op, self._classifier_loss_t], feed_dict)
        return loss

    def _epoch_after_hook(self, *args, **kwargs):
        """Hook called at the end of each epoch."""
        #TODO Avi remove the 1000 and put in a parameter for it
        for i in range(self._n_classifier_train_steps_update):
            self._train_classifier_step()
        #import pdb; pdb.set_trace()

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        diagnostics = super(SACClassifier, self).get_diagnostics(
            iteration, batch, training_paths, evaluation_paths)
        
        sample_observations = batch['observations']
        goal_index = np.random.randint(self._goal_examples.shape[0],
                                       size=sample_observations.shape[0])
        goal_observations = self._goal_examples[goal_index]

        sample_goal_observations = np.concatenate(
            (sample_observations, goal_observations), axis=0)

        reward_sample_goal_observations, classifier_loss = self._session.run(
            [self._reward_t, self._classifier_loss_t],
            feed_dict={self._observations_ph: sample_goal_observations,
                       self._label_ph: np.concatenate([
                                    np.zeros((sample_observations.shape[0],1)),
                                    np.ones((goal_observations.shape[0],1)),
                                    ])
                        }
            )

        reward_sample_observations, reward_goal_observations = np.split(
            reward_sample_goal_observations,
            (sample_observations.shape[0],),
            axis=0)

        diagnostics.update({
            'reward_learning/classifier_loss': np.mean(classifier_loss),
            'reward_learning/reward_sample_obs_mean': np.mean(
                reward_sample_observations),
            'reward_learning/reward_goal_obs_mean': np.mean(
                reward_goal_observations),
        })

        return diagnostics