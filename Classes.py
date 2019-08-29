import numpy as np
import keras.backend as K
import tensorflow as tf
import tensorflow.contrib.slim as slim

from keras.models import Model
from keras import regularizers
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Flatten, LSTM, LSTMCell
from keras.optimizers import Adam


class Network:
    def __init__(self, scope, act_dim, num_units, input_size, gamma = 0.99, lr = 0.0001, optimizer='rmsprop'):
        """ Initialization
        """
        # set general parameters
        self.gamma = gamma
        self.learning_rate = lr
        self.optimizer = optimizer
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.state = K.placeholder(shape=[None, 1, input_size], dtype=tf.float32)
            self.prev_rewards = K.placeholder(shape=[None, 1], dtype=tf.float32)
            self.prev_actions = K.placeholder(shape=[None], dtype=tf.int32)
            self.timestep = K.placeholder(shape=[None, 1], dtype=tf.float32)
            self.prev_actions_one_hot = tf.one_hot(self.prev_actions, act_dim, dtype=tf.float32)
            self.actions = K.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, act_dim, dtype=tf.float32)

            hidden = tf.concat([slim.flatten(self.state), self.prev_rewards, self.prev_actions_one_hot, self.timestep],
                               1)
            lstm_cell = LSTMCell(num_units)

            self.stat_init = [np.zeros((1, num_units), np.float32), np.zeros((1, num_units), np.float32)]
            self.state_in = [tf.placeholder(tf.float32, [1, num_units]), tf.placeholder(tf.float32, [1, num_units])]

            step_size = tf.shape(self.prev_rewards)[:1]

            input_layer = Input(shape=(1, input_size + 1 + act_dim)) # Network input_shape - state + reward + one_hot action rep
            lstm_layer = LSTM(num_units)(input_layer)
            dense_layer = Dense(num_units)(lstm_layer)
            self.model = Model(input_layer, dense_layer)

            #Output layers for policy and value estimations

            policy_layer = Dense(act_dim, activation='softmax')(self.model.output)
            value_layer = Dense(1, activation='linear')(self.model.output)

            self.policy = Model(input_layer, policy_layer)
            self.value = Model(input_layer, value_layer)
            # self.model.add(lstm_layer)
            # lstm_c, lstm_h = lstm_state
            # self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            # rnn_out = tf.reshape(lstm_outputs, [-1, 48])

           # # Todo: from here on
           #  # Only the worker network need ops for loss functions and gradient updating.
           #  if scope != 'global':
           #      self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
           #      self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
           #
           #      self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
           #
           #      # Loss functions
           #      self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
           #      self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
           #      self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs + 1e-7) * self.advantages)
           #      self.loss = 0.05 * self.value_loss + self.policy_loss - self.entropy * 0.05
           #
           #      # Get gradients from local network using local losses
           #      local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
           #      self.gradients = tf.gradients(self.loss, local_vars)
           #      self.var_norms = tf.global_norm(local_vars)
           #      grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 999.0)
           #
           #      # Apply local gradients to global network
           #      global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
           #      self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))

    def train(self, hidden_representations, labels, good_label, episode_action, episode_reward, reward, a, t):
        a =1

    def select_action(self, state):
        """use the policy network to select the next action to take """
        state = np.expand_dims(state,0)
        action_probs = self.policy.predict(state)
        selected_action = np.random.choice(action_probs.size, 1, p=action_probs.ravel())
        return selected_action

    # def update_state(self):






