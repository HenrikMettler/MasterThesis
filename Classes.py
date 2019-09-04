import numpy as np
import keras.backend as K
import tensorflow as tf
import tensorflow.contrib.slim as slim
import datetime

from keras import regularizers
from keras.utils import to_categorical
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Flatten, LSTM
from keras.optimizers import RMSprop

class Network:
    def __init__(self, scope, act_dim, num_units, input_size, gamma=0.99, lr=0.0001):
        """ Initialization
        """
        # set general parameters
        self.gamma = gamma
        self.learning_rate = lr
        self.optimizer = RMSprop(lr=lr, epsilon=0.1, rho=0.99)

        # Input and visual encoding layers
        self.prev_rewards = K.placeholder(shape=[None, 1], dtype=tf.float32)
        self.prev_states = K.placeholder(shape=[None, 1, input_size], dtype=tf.float32)

        # self.state = K.placeholder(shape=[None, 1, input_size], dtype=tf.float32)
        # self.prev_rewards = K.placeholder(shape=[None, 1], dtype=tf.float32)
        # self.prev_actions = K.placeholder(shape=[None], dtype=tf.int32)
        # #self.timestep = K.placeholder(shape=[None, 1], dtype=tf.float32)
        # self.prev_actions_one_hot = tf.one_hot(self.prev_actions, act_dim, dtype=tf.float32)
        # self.actions = K.placeholder(shape=[None], dtype=tf.int32)
        # self.actions_onehot = tf.one_hot(self.actions, act_dim, dtype=tf.float32)

        #hidden = tf.concat([slim.flatten(self.state), self.prev_rewards, self.prev_actions_one_hot, self.timestep],1)
            #lstm_cell = LSTMCell(num_units)

        self.stat_init = [np.zeros((1, num_units), np.float32), np.zeros((1, num_units), np.float32)]
        self.state_in = [tf.placeholder(tf.float32, [1, num_units]), tf.placeholder(tf.float32, [1, num_units])]

        step_size = tf.shape(self.prev_rewards)[:1]

        input_layer = Input(shape=(1, input_size )) # Network input_shape - state + reward + one_hot action rep
        lstm_layer = LSTM(num_units)(input_layer)
        dense_layer = Dense(num_units)(lstm_layer)
        self.model = Model(input_layer, dense_layer)

        # Output layers for policy and value estimations
        policy_layer = Dense(act_dim, activation='softmax')(self.model.output)
        value_layer = Dense(1, activation='linear')(self.model.output)

        self.policy_model = Model(input_layer, policy_layer)
        self.value_model = Model(input_layer, value_layer)

        # def value_loss_function(y_true, y_pred):
        #     """define the loss function for the value network"""
        #     #compute discounted rewards
        #     discounted_rewards = self.discount_rewards(self.prev_rewards, self.gamma)
        #
        #     # calculate the loss of the critic
        #     critic_loss = np.mean(np.square(discounted_rewards-y_pred))
        #
        #     return critic_loss

        self.value_model.compile(optimizer=self.optimizer, loss='mean_squared_error')

        def policy_loss_function(y_true, y_pred):
            """define the loss function for the policy"""

            advantages = y_true # As a reminder of the way the policy gradient was implemented
            out = K.clip(y_pred, 1e-8, 1-1e-8) # clip the prediction values before calculating the loss function

            log_likelihood = K.log(out)


            loss = K.sum(-log_likelihood*advantages)
            return loss  # gradient ascent not descent

        self.policy_model.compile(optimizer=self.optimizer, loss=policy_loss_function)



    def train(self, episode_states, episode_actions, episode_rewards, episode_logprobs):
        """Train the Policy (Actor) and Value (Critic) based on discounted rewards"""

        # compute discounted rewards
        discounted_rewards = self.discount_rewards(episode_rewards, self.gamma)

        # compute state value predictions
        #input_states = np.concatenate([episode_states, episode_rewards, episode_actions],2)
        state_values = self.value_model.predict(episode_states)

        # compute advantages
        advantages = discounted_rewards - state_values

        # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


        # optimize the network
        self.value_model.fit(episode_states, y=discounted_rewards)#, callbacks=[tensorboard_callback]) # the discounted rewards are the target values for the Critic
        # Todo: check about other parameters in this fit method!
        self.policy_model.fit(episode_states, y=advantages)#, callbacks=[tensorboard_callback]) # "abuse" the advantages as labels, for the custom loss function
    # def policy_loss_function(self, advantages, logprobs):
    #     log_times_advantages = advantages* logprobs
    #     j = np.sum(log_times_advantages)
    #     return -j # gradient ascent not descent

    def optimize_policy(self, episode_actions, episode_states, advantages, episode_logprobs):
        """ Optimize the policy network
        Formula: Nabla-params = Sum_over_time (Nabla log(policy(a_t|s_t, network_params)*H(t)) || H(t) = advantages"""
        log_times_advantages = advantages*episode_logprobs
        j = np.sum(log_times_advantages)

        # calculate weighted actions
        # eligibility = K.log(weighted_actions + 1e-10) * K.stop_gradient(self.advantages_pl)
        # entropy = K.sum(self.model.output * K.log(self.model.output + 1e-10), axis=1)
        # loss = 0.001 * entropy - K.sum(eligibility)

        #     updates = self.rms_optimizer.get_updates(self.model.trainable_weights, [], loss)
        #     return K.function([self.model.input, self.action_pl, self.advantages_pl], [], updates=updates)

    def optimize_value(self, state_values, discounted_rewards):
        """ Optimize the Value network"""
        critic_loss = np.mean(np.square(discounted_rewards-state_values))
        updates = self.optimizer.get_updates(self.model.trainable_weights, critic_loss)

        return K.function([self.value_model, discounted_rewards], [], updates=updates)

        # """ Critic Optimization: Mean Squared Error over discounted rewards
        #         """
        # critic_loss = K.mean(K.square(self.discounted_r - self.model.output))
        # updates = self.rms_optimizer.get_updates(self.model.trainable_weights, [], critic_loss)
        # return K.function([self.model.input, self.discounted_r], [], updates=updates)


    def discount_rewards (self, episode_rewards, gamma):
        """Discount rewards over an episode, using gamma"""
        discounted_rewards = np.zeros([np.size(episode_rewards,0),1])
        cumulated_rewards = 0
        for time in reversed(range(0, len(episode_rewards))):
            cumulated_rewards = episode_rewards[time] + cumulated_rewards*gamma
            discounted_rewards[time] = cumulated_rewards
        return discounted_rewards

    def select_action(self, state):
        """use the policy network to select the next action to take """

        state = np.expand_dims(state,0)
        action_probs = self.policy_model.predict(state)
        selected_action = np.random.choice(action_probs.size, 1, p=action_probs.ravel())
        log_prob = np.log(action_probs.squeeze(0)[selected_action])

        return selected_action, log_prob

    # def update_state(self):






