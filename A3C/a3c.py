# Source: https://github.com/germain-hug/Deep-RL-Keras#n-step-asynchronous-advantage-actor-critic-a3c

import sys
import gym
import time
import threading
import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim

from tqdm import tqdm
from keras.models import Model, Sequential
from keras import regularizers
from keras.layers import Input, LSTM, LSTMCell, Dense, Flatten, Reshape, RNN
from keras.optimizers import RMSprop
from keras import backend as K
from tensorflow.contrib import rnn as rnn
from functions import *

from .critic import Critic
from .actor import Actor
from .thread import training_thread
from utils.continuous_environments import Environment
from utils.networks import conv_block
from utils.stats import gather_stats

class A3C:
    """ Asynchronous Actor-Critic Main Algorithm
    """

    def __init__(self, act_dim, env_dim, k, lstm_param_list, dataset_size, gamma = 0.99, lr = 0.0007, optimizer=RMSprop, n_timeSteps=100):
        """ Initialization
        """
        # Environment and A3C parameters
        self.act_dim = act_dim
        self.env_dim = env_dim #(k,) + env_dim #Todo: understand this
        self.gamma = gamma
        self.lr = lr
        self.optimizer = optimizer
        self.n_timeSteps = n_timeSteps
        # data set size
        self.dataset_size = dataset_size
        # Create actor and critic networks
        self.lstm_param_list = lstm_param_list
        self.shared = self.buildNetwork()
        self.actor = Actor(self.env_dim, act_dim, self.shared, lr)
        self.critic = Critic(self.env_dim, act_dim, self.shared, lr)
        # Build optimizers
        self.a_opt = self.actor.optimizer()
        self.c_opt = self.critic.optimizer()

    def buildNetwork(self):

        # define placeholder variables for input, encoding layers
        state = K.placeholder(shape=[None, 2, self.dataset_size])
        prev_reward = K.placeholder(shape=[None,1])
        prev_actions = K.placeholder(shape=[None], dtype=tf.int32)
        prev_actions_onehot = K.one_hot(prev_actions, self.act_dim) # ToDo: change to action space dim
        timestep = K.placeholder(shape=[None,1])
        hidden = K.concatenate([slim.flatten(state), prev_reward, prev_actions_onehot, timestep], 1)

        # initialize Lstm cells
        lstm_cells = LSTMCell(units=self.lstm_param_list[0], activation=self.lstm_param_list[1],
                              recurrent_activation=self.lstm_param_list[2], use_bias=self.lstm_param_list[3],
                              kernel_initializer=self.lstm_param_list[4], recurrent_initializer=self.lstm_param_list[5],
                              bias_initializer=self.lstm_param_list[6], unit_forget_bias=self.lstm_param_list[7],
                              kernel_regularizer=self.lstm_param_list[8], recurrent_regularizer=self.lstm_param_list[9],
                              bias_regularizer=self.lstm_param_list[10], kernel_constraint=self.lstm_param_list[11],
                              recurrent_constraint=self.lstm_param_list[12], bias_constraint=self.lstm_param_list[13],
                              dropout=self.lstm_param_list[14], recurrent_dropout=self.lstm_param_list[15],
                              implementation=self.lstm_param_list[16])
        # initialize cell state
        c_init = np.zeros((1,lstm_cells.units))
        h_init = np.zeros((1,lstm_cells.units))
        state_init = [c_init,h_init]
        c_in = K.placeholder([1,lstm_cells.units])
        h_in = K.placeholder([1,lstm_cells.units])
        state_in = (c_in, h_in)
        lstm_in = K.expand_dims(hidden,[0])
        step_size = K.shape(prev_reward)[:1]
        state_tuple = rnn.LSTMStateTuple(c_in, h_in)
        lstm_layer = RNN(lstm_cells, lstm_in, return_state='true')
        #lstm_c, lstm_h = lstm_state
        #state_out = (lstm_c[:1, :], lstm_h[:1, :])
        #lstm_output = tf.reshape(lstm_out, [-1, 48])
        actions = K.placeholder(shape=[None], dtype=tf.int32)
        actions_onehot = K.one_hot(actions, self.act_dim)

        policy = Dense(256, self.act_dim, activation='softmax', weights=normalized_columns_initializer(0.01))
        value = Dense(256, 1, activation=None, weights=normalized_columns_initializer(1.0))


        #inp = Input(self.env_dim)

        # lstm_layer = LSTM(units=self.lstm_param_list[0], activation=self.lstm_param_list[1],
        #                     recurrent_activation=self.lstm_param_list[2], use_bias=self.lstm_param_list[3],
        #                     kernel_initializer=self.lstm_param_list[4], recurrent_initializer=self.lstm_param_list[5],
        #                     bias_initializer=self.lstm_param_list[6], unit_forget_bias=self.lstm_param_list[7],
        #                     kernel_regularizer=self.lstm_param_list[8], recurrent_regularizer=self.lstm_param_list[9],
        #                     bias_regularizer=self.lstm_param_list[10], activity_regularizer=self.lstm_param_list[17],
        #                     kernel_constraint=self.lstm_param_list[11],
        #                     recurrent_constraint=self.lstm_param_list[12], bias_constraint=self.lstm_param_list[13],
        #                     dropout=self.lstm_param_list[14], recurrent_dropout=self.lstm_param_list[15],
        #                     implementation=self.lstm_param_list[16], return_sequences=self.lstm_param_list[18],
        #                     return_state=self.lstm_param_list[19])
        #
        # model = Sequential()
        # model.add(lstm_layer,input_shape = [self.env_dim,])
        # #Model(inp, lstm_layer, Dense(1,))
        #
        # model.compile(optimzer=self.optimizer)

        # Todo: cast outputs into one struct
        return policy, value, actions, actions_onehot, lstm_state, state_in, state_init, state, prev_reward, \
               prev_actions, timestep, prev_actions_onehot


    def policy_action(self, s):
        """ Use the actor's network to predict the next action to take, using the policy
        """
        return np.random.choice(np.arange(self.act_dim), 1, p=self.actor.predict(s).ravel())[0]

    def discount(self, r, done, s):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def train_models(self, states, actions, rewards, done):
        """ Update actor and critic networks from experience
        """
        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = self.discount(rewards, done, states[-1])
        state_values = self.critic.predict(np.array(states))
        advantages = discounted_rewards - np.reshape(state_values, len(state_values))
        # Networks optimization
        self.a_opt([states, actions, advantages])
        self.c_opt([states, discounted_rewards])

    def train(self, args, summary_writer):

        # Instantiate one environment per thread
        envs = [Environment(gym.make(args.env), args.consecutive_frames) for i in range(args.n_threads)]
        [e.reset() for e in envs]
        state_dim = envs[0].get_state_size()
        action_dim = gym.make(args.env).action_space.n
        # Create threads
        tqdm_e = tqdm(range(int(args.nb_episodes)), desc='Score', leave=True, unit=" episodes")

        threads = [threading.Thread(
                target=training_thread,
                daemon=True,
                args=(self,
                    args.nb_episodes,
                    envs[i],
                    action_dim,
                    args.training_interval,
                    summary_writer,
                    tqdm_e,
                    args.render)) for i in range(args.n_threads)]

        for t in threads:
            t.start()
            time.sleep(0.5)
        try:
            [t.join() for t in threads]
        except KeyboardInterrupt:
            print("Exiting all threads...")
        return None

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)
