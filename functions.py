# imports
import numpy as np

from TwoStepTask import *
from keras.layers import LSTM, LSTMCell, Dense
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import RMSprop


def create_environment(environment_name):
    if environment_name == 'twoStepTask':
        environment = TwoStepTask()
    #elif environment_name == 'harlow':
        environment = 1
    else:
        raise NameError('Environment does not exist yet, please use harlow or two step task')
    return environment


def create_dm_network(lstm_param_list, optimizer):
    # Todo: Why are the params in cell and model??
    lstm_cells = LSTMCell(num_units=lstm_param_list[0], activation=lstm_param_list[1],
                          recurrent_activation=lstm_param_list[2], use_bias=lstm_param_list[3],
                          kernel_initializer=lstm_param_list[4], recurrent_initializer=lstm_param_list[5],
                          bias_initializer=lstm_param_list[6], unit_forget_bias=lstm_param_list[7],
                          kernel_regularizer=lstm_param_list[8], recurrent_regularizer=lstm_param_list[9],
                          bias_regularizer=lstm_param_list[10], kernel_constraint=lstm_param_list[11],
                          recurrent_constraint=lstm_param_list[12], bias_constraint=lstm_param_list[13],
                          dropout=lstm_param_list[14], recurrent_dropout=lstm_param_list[15],
                          implementation=lstm_param_list[16])

    lstm_network = LSTM(lstm_cells, num_units = lstm_param_list[0],activation=lstm_param_list[1],
                        recurrent_activation=lstm_param_list[2], use_bias=lstm_param_list[3],
                        kernel_initializer=lstm_param_list[4], recurrent_initializer=lstm_param_list[5],
                        bias_initializer=lstm_param_list[6], unit_forget_bias=lstm_param_list[7],
                        kernel_regularizer=lstm_param_list[8], recurrent_regularizer=lstm_param_list[9],
                        bias_regularizer=lstm_param_list[10], activity_regularizer = lstm_param_list[17],
                        kernel_constraint=lstm_param_list[11],
                        recurrent_constraint=lstm_param_list[12], bias_constraint=lstm_param_list[13],
                        dropout=lstm_param_list[14], recurrent_dropout=lstm_param_list[15],
                        implementation=lstm_param_list[16], return_sequences = lstm_param_list[18],
                        return_state=lstm_param_list[19])

    dm_network = Sequential()
    # Todo: This probably lacks proper input layer and output layer structure
    dm_network.add(lstm_network)
    dm_network.compile(optimizer=optimizer)
    dm_network.summary()

    return dm_network


def create_worker():
    meta_rl_worker = 1
    return meta_rl_worker


def training(trainer = 0, dm_network = 0, gamma = 0):
    updated_trainer = trainer + gamma * dm_network
    return updated_trainer


class Create_Args():
    # class for the implementation of an args data structure
    def __init__(self,nb_episodes, batch_size, consecutive_frames, training_interval, n_threads, gamma, lr, gather_stats=0, render=0, env='empty'):
        self.nb_episode = nb_episodes
        self.batch_size = batch_size
        self.consecutive_frames = consecutive_frames
        self.training_interval = training_interval
        self.n_threads = n_threads
        self.gamma = gamma
        self.lr = lr
        self.gather_stats = gather_stats
        self.render = render
        self.env = env


