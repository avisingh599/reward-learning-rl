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
            classifier_lr=1e-4, #TODO add this to variants/ExerimentRunner/utils
            classifier_batch_size=128, #TODO add this to variant
            n_classifier_train_steps=int(1e4),
            classifier_optim_name='adam',
            **kwargs,
    ):
        
        self._classifier = classifier
        self._goal_examples = goal_examples
        self._classifier_lr = classifier_lr
        self._n_classifier_train_steps = n_classifier_train_steps
        self._classifier_optim_name = classifier_optim_name
        self._classifier_batch_size = classifier_batch_size

        super(SACClassifier, self).__init__(**kwargs)

    def _build(self):
        self._training_ops = {}

        self._init_global_step()
        self._init_placeholders()
        self._init_classifier_update()
        self._init_actor_update()
        self._init_critic_update()

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

        # rewards = self._classifier(self._observations_ph)[:,0]
        #rewards = self._discriminator_t

        Q_target = td_target(
            reward=self._reward_scale * self._reward_t,
            discount=self._discount,
            next_value=(1 - self._terminals_ph) * next_value)

        return Q_target

    def _init_classifier_update(self):
        #import pdb; pdb.set_trace()
        #TODO Avi modify this to match whatever we do in VICE
        logits = self._classifier([self._observations_ph])
        #logits = tf.expand_dims(logits, axis=-1)
        # self._discriminator_t = tf.squeeze(tf.nn.sigmoid(logits), axis=-1)
        self._discriminator_t = tf.nn.sigmoid(logits)

        #classifier_logprob = -1*tf.nn.relu(energy)
        #classifier_logprob = -10*tf.nn.sigmoid(classifier_output_linear)
        
        cross_entropy_t = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=self._label_ph)
        self._loss_t = tf.reduce_mean(cross_entropy_t)
        self._reward_t = self._discriminator_t

        # self._classifier_train_op = tf.train.AdamOptimizer(self._classifier_lr).minimize(
        #         loss=self._loss_t, var_list=self._classifier.get_params_internal())

        if self._classifier_optim_name == 'adam':
            self._classifier_optimizer = tf.train.AdamOptimizer(
                                        learning_rate=self._classifier_lr,
                                        name='classifier_optimizer')
        else:
            raise NotImplementedError

        self._classifier_training_op = \
            tf.contrib.layers.optimize_loss(
                self._loss_t,
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
        self._session.run(self._classifier_training_op, feed_dict)
    
    def _epoch_after_hook(self, *args, **kwargs):
        """Hook called at the end of each epoch."""
        #TODO Avi remove the 1000 and put in a parameter for it
        for i in range(1000):
            self._train_classifier_step()

        #pass

    # def _get_Q_target(self):
    #     #TODO Avi make sure it uses self._reward_t instead of 
    #     next_actions = self._policy.actions([self._next_observations_ph])
    #     next_log_pis = self._policy.log_pis(
    #         [self._next_observations_ph], next_actions)

    #     next_Qs_values = tuple(
    #         Q([self._next_observations_ph, next_actions])
    #         for Q in self._Q_targets)

    #     min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
    #     next_value = min_next_Q - self._alpha * next_log_pis

    #     Q_target = td_target(
    #         reward=self._reward_scale * self._rewards_ph,
    #         discount=self._discount,
    #         next_value=(1 - self._terminals_ph) * next_value)

    #     return Q_target
