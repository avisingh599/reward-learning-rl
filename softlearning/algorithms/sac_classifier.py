import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util

from .sac import SAC, td_target

class SACClassifier(SAC):
    def __init__(
            self,
            classifier,
            goal_examples,
            goal_examples_validation,
            classifier_lr=1e-4,
            classifier_batch_size=128,
            reward_type = 'logits',
            n_classifier_train_steps=int(1e4),
            classifier_optim_name='adam',
            **kwargs,
    ):
        
        self._classifier = classifier
        self._goal_examples = goal_examples
        self._goal_examples_validation = goal_examples_validation
        self._classifier_lr = classifier_lr
        self._reward_type = reward_type
        self._n_classifier_train_steps = n_classifier_train_steps
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
        if self._reward_type == 'logits':
            self._reward_t = observation_logits
        elif self._reward_type == 'probabilities':
            self._reward_t = tf.nn.sigmoid(observation_logits)
        else:
            raise NotImplementedError('Unknown reward type: {}'.format(self._reward_type))

        Q_target = td_target(
            reward=self._reward_scale * self._reward_t,
            discount=self._discount,
            next_value=(1 - self._terminals_ph) * next_value)

        return Q_target

    def _get_classifier_training_op(self):
        if self._classifier_optim_name == 'adam':
            opt_func = tf.train.AdamOptimizer
        elif self._classifier_optim_name == 'sgd':
            opt_func = tf.train.GradientDescentOptimizer
        else:
            raise NotImplementedError

        self._classifier_optimizer = opt_func(
            learning_rate=self._classifier_lr,
            name='classifier_optimizer')

        classifier_training_op = \
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

        return classifier_training_op

    def _init_classifier_update(self):
        logits = self._classifier([self._observations_ph])
        self._classifier_loss_t = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=self._label_ph))
        self._classifier_training_op = self._get_classifier_training_op()


    def _get_classifier_feed_dict(self):

        negatives = self.sampler.random_batch(self._classifier_batch_size)['observations']
        rand_positive_ind = np.random.randint(self._goal_examples.shape[0], size=self._classifier_batch_size)
        positives = self._goal_examples[rand_positive_ind]

        labels_batch = np.zeros((2*self._classifier_batch_size,1))
        labels_batch[self._classifier_batch_size:] = 1.0
        observation_batch = np.concatenate([negatives, positives], axis=0)

        from softlearning.misc.utils import mixup
        observation_batch, labels_batch = mixup(observation_batch, labels_batch)
        feed_dict = {
            self._observations_ph: observation_batch,
            self._label_ph: labels_batch
        }

        return feed_dict


    def _train_classifier_step(self, feed_dict):
        _, loss = self._session.run([self._classifier_training_op, self._classifier_loss_t], feed_dict)
        return loss

    def _epoch_after_hook(self, *args, **kwargs):
        #import IPython; IPython.embed()
        if self._epoch == 0:
            for i in range(self._n_classifier_train_steps):
                feed_dict = self._get_classifier_feed_dict()
                self._train_classifier_step(feed_dict)

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
        
        goal_index_validation = np.random.randint(
            self._goal_examples_validation.shape[0],
            size=sample_observations.shape[0])
        goal_observations_validation = \
            self._goal_examples_validation[goal_index_validation]

        sample_goal_observations = np.concatenate(
            (sample_observations, goal_observations, goal_observations_validation),
            axis=0)

        reward_sample_goal_observations, classifier_loss = self._session.run(
            [self._reward_t, self._classifier_loss_t],
            feed_dict={self._observations_ph: sample_goal_observations,
                       self._label_ph: np.concatenate([
                                    np.zeros((sample_observations.shape[0],1)),
                                    np.ones((goal_observations.shape[0],1)),
                                    np.ones((goal_observations_validation.shape[0],1)),
                                    ])
                        }
            )

        #TODO Avi Make this clearer. Maybe just make all the vectors
        #the same size and specify number of splits
        reward_sample_observations, reward_goal_observations, \
        reward_goal_observations_validation = np.split(
            reward_sample_goal_observations,
            (sample_observations.shape[0],
             sample_observations.shape[0]+goal_observations.shape[0]
            ),
            axis=0)
        
        #TODO Avi fix this so that classifier loss is split into train and val
        #currently the classifier loss printed is the mean
        # classifier_loss_train, classifier_loss_validation = np.split(
        #     classifier_loss,
        #     (sample_observations.shape[0]+goal_observations.shape[0],),
        #     axis=0)

        diagnostics.update({
            #'reward_learning/classifier_loss_train': np.mean(classifier_loss_train),
            #'reward_learning/classifier_loss_validation': np.mean(classifier_loss_validation),
            'reward_learning/classifier_loss': classifier_loss,
            'reward_learning/reward_sample_obs_mean': np.mean(
                reward_sample_observations),
            'reward_learning/reward_goal_obs_mean': np.mean(
                reward_goal_observations),
            'reward_learning/reward_goal_obs_validation_mean': np.mean(
                reward_goal_observations_validation),
        })

        return diagnostics

    def _evaluate_rollouts(self, paths, env):
        """Compute evaluation metrics for the given rollouts."""
        diagnostics = super(SACClassifier, self)._evaluate_rollouts(paths, env)
        
        observations = [path['observations'] for path in paths]
        observations = np.concatenate(observations)
        learned_reward = self._session.run(self._reward_t,
            feed_dict={self._observations_ph: observations})

        diagnostics[f'reward_learning/reward-mean'] = np.mean(learned_reward)
        diagnostics[f'reward_learning/reward-min'] = np.min(learned_reward)
        diagnostics[f'reward_learning/reward-max'] = np.max(learned_reward)
        diagnostics[f'reward_learning/reward-std'] = np.std(learned_reward)

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = super(SACClassifier, self).tf_saveables
        saveables.update({
            '_classifier_optimizer': self._classifier_optimizer
        })

        return saveables