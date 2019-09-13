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

        # Placeholder variables
        self.action_pl = K.placeholder(shape=(None, act_dim))
        self.advantages = K.placeholder(shape=(None,))
        self.discounted_rewards = K.placeholder(shape=(None,))

        # Input and visual encoding layers
        self.prev_rewards = K.placeholder(shape=[None, 1], dtype=tf.float32)
        self.prev_states = K.placeholder(shape=[None, 1, input_size], dtype=tf.float32)

        input_layer = Input(shape=(1, input_size ))
        lstm_layer = LSTM(num_units)(input_layer)
        dense_layer = Dense(num_units)(lstm_layer)
        self.model = Model(input_layer, dense_layer)

        # Output layers for policy and value estimations
        policy_layer = Dense(act_dim, activation='softmax')(self.model.output)
        value_layer = Dense(1, activation='linear')(self.model.output)

        self.policy_model = Model(input_layer, policy_layer)
        self.value_model = Model(input_layer, value_layer)

        # Build optimizers
        self.policy_optimizer = self.build_policy_optimizer()
        self.value_optimizer = self.build_value_optimizer()


        # self.value_model.compile(optimizer=self.optimizer, loss='mean_squared_error')
        #
        # def policy_loss_function(y_true, y_pred):
        #     """define the loss function for the policy"""
        #
        #     advantages = y_true # As a reminder of the way the policy gradient was implemented
        #     out = K.clip(y_pred, 1e-8, 1-1e-8) # clip the prediction values before calculating the loss function
        #
        #     log_likelihood = K.log(out)
        #
        #
        #     loss = K.sum(-log_likelihood*advantages)
        #     return -loss  # gradient ascent not descent
        #
        # self.policy_model.compile(optimizer=self.optimizer, loss=policy_loss_function)


    def build_policy_optimizer(self):

        weighted_actions = K.sum(self.action_pl*self.policy_model.output, axis=1)
        eligibility = K.log(weighted_actions + 1e-10)*K.stop_gradient(self.advantages)

        entropy = K.sum(self.policy_model.output * K.log(self.policy_model.output+ 1e-10))
        loss = 0.001 * entropy - K.sum(eligibility)

        #self.optimizer.get_updates()

        updates = self.optimizer.get_updates(loss=loss, params=self.policy_model.trainable_weights)
        return K.function([self.policy_model.input, self.action_pl, self.advantages], [], updates=updates)

        # def optimizer(self):
        #     """ Actor Optimization: Advantages + Entropy term to encourage exploration
        #     (Cf. https://arxiv.org/abs/1602.01783)
        #     """
        #     weighted_actions = K.sum(self.action_pl * self.model.output, axis=1)
        #     eligibility = K.log(weighted_actions + 1e-10) * K.stop_gradient(self.advantages_pl)
        #     entropy = K.sum(self.model.output * K.log(self.model.output + 1e-10), axis=1)
        #     loss = 0.001 * entropy - K.sum(eligibility)
        #
        #     updates = self.rms_optimizer.get_updates(self.model.trainable_weights, [], loss)
        #     return K.function([self.model.input, self.action_pl, self.advantages_pl], [], updates=updates)

    def build_value_optimizer(self):

        critic_loss = K.mean(K.square(self.discounted_rewards - self.value_model.output))
        updates = self.optimizer.get_updates(critic_loss, self.value_model.trainable_weights)

        return K.function([self.value_model.input, self.discounted_rewards], [], updates=updates)

        # def optimizer(self):
        #     """ Critic Optimization: Mean Squared Error over discounted rewards
        #     """
        #     critic_loss = K.mean(K.square(self.discounted_r - self.model.output))
        #     updates = self.rms_optimizer.get_updates(self.model.trainable_weights, [], critic_loss)
        #     return K.function([self.model.input, self.discounted_r], [], updates=updates)

    def train(self, episode_states, episode_actions, episode_rewards, episode_logprobs):
        """Train the Policy (Actor) and Value (Critic) based on discounted rewards"""

        # compute discounted rewards
        discounted_rewards = self.discount_rewards(episode_rewards, self.gamma)

        # compute state value predictions
        state_values = self.value_model.predict(episode_states)

        # compute advantages
        advantages = discounted_rewards - state_values

        # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


        # optimize the network
        self.policy_optimizer([episode_actions, advantages])
        self.value_optimizer([episode_states, discounted_rewards])

        #self.value_model.fit(episode_states, y=discounted_rewards, verbose=0)#, callbacks=[tensorboard_callback]) # the discounted rewards are the target values for the Critic

        #
        #self.policy_model.fit(episode_states, y=advantages, verbose=0)#, callbacks=[tensorboard_callback]) # "abuse" the advantages as labels, for the custom loss function

            # def train_models(self, states, actions, rewards, done):
            # """ Update actor and critic networks from experience
            # """
            # # Compute discounted rewards and Advantage (TD. Error)
            # discounted_rewards = self.discount(rewards)
            # state_values = self.critic.predict(np.array(states))
            # advantages = discounted_rewards - np.reshape(state_values, len(state_values))
            # # Networks optimization
            # self.a_opt([states, actions, advantages])
            # self.c_opt([states, discounted_rewards])

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


class Datastructure:
    def __init__(self,save_states, save_actions, save_rewards, save_trialidx, save_labels, save_goodlabels, save_hiddenreps):
        self.state = save_states
        self.actions = save_actions
        self.rewards = save_rewards
        self.trialidx = save_trialidx
        self.labels = save_labels
        self.goodlabels = save_goodlabels
        self.hidden_reps = save_hiddenreps