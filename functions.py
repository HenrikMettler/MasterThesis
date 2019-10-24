import os
import numpy as np
# import pickle
from keras.layers import LSTM, LSTMCell, Dense
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import RMSprop


def create_sample_representation(encoder_Model, data, label, is_training):
    # pick up two random elements from the data set and their labels
    size_data = label.size
    element_one = np.random.randint(0,size_data-1)
    element_two = np.random.randint(0,size_data-1)
    label_one = label[element_one]
    label_two = label[element_two]

    # pick up the data from the element one
    data_one = data[element_one]
    picked_data = np.zeros((2, np.shape(data_one)[0], np.shape(data_one)[1], np.shape(data_one)[2]))
    picked_data[0,:,:,:] = data_one

    # check that they belong to different classes, if not repeat replacing the 2nd until the labels are no longer equal
    if label_one == label_two:
        label_are_equal = 'true'
    else:
        label_are_equal = 'false'

    while label_are_equal == 'true':
        element_two = np.random.randint(0,size_data-1)
        label_two = label[element_two]
        if label_one != label_two:
            label_are_equal = 'false'

    # pick up the data for the second element
    data_two = data[element_two]
    picked_data [1,:,:,:] = data_two

    # calculate the hidden representation of the two data inputs
    hidden_representations_beforereshape = encoder_Model.predict(picked_data)
    # reshape the hidden representations
    hidden_rep_1 = hidden_representations_beforereshape[0, :, :, :]
    hidden_rep_1 = hidden_rep_1.reshape([1, np.shape(hidden_rep_1)[0] * np.shape(hidden_rep_1)[1] * np.shape(hidden_rep_1)[2]])
    hidden_rep_2 = hidden_representations_beforereshape[1, :, :, :]
    hidden_rep_2 = hidden_rep_2.reshape([1, np.shape(hidden_rep_2)[0] * np.shape(hidden_rep_2)[1] * np.shape(hidden_rep_2)[2]])
    # put into one array
    hidden_representations = np.zeros((2, np.shape(hidden_rep_1,)[1]))
    hidden_representations[0, :] = hidden_rep_1
    hidden_representations[1, :] = hidden_rep_2

    # return the labels as a vector
    labels_out = [label_one, label_two]

    return hidden_representations, labels_out # explicitly don't return the data, as it is not accessible to the DM!


def create_dm_network(num_units, input_shape, learning_rate, loss, output_activation='sigmoid'):


    lstm_layer = LSTM(num_units, return_sequences=True, input_shape=(1, input_shape))

    dm_network = Sequential()
    dm_network.add(lstm_layer)
    dm_network.add(Dense(1,activation=output_activation))
    optimizer = RMSprop(learning_rate)
    dm_network.compile(optimizer=optimizer, loss=loss)
    dm_network.summary()

    return dm_network


def create_networks(num_units, input_shape, optimizer, loss, output_activation='sigmoid', time_step_per_trial = 1):
    lstm_layer = LSTM(num_units,return_sequences=True, input_shape=(time_step_per_trial, input_shape))

    action_network = Sequential()
    action_network.add(lstm_layer)
    action_network.add(Dense(1,activation=output_activation))
    action_network.compile(optimizer=optimizer, loss=loss)
    action_network.summary()

    value_network = Sequential()
    value_network.add(lstm_layer)
    value_network.add(Dense(1,activation=output_activation))
    value_network.compile(optimizer=optimizer, loss=loss)
    value_network.summary()

    return action_network, value_network


def mix_up(hidden_rep1, hidden_rep2, labels):
    label1 = labels[0]
    label2 = labels[1]
    #copy_labels = labels
    p = np.random.random_sample()

    hidden_representations = np.zeros((2, np.size(hidden_rep1)))
    # change the order if p > 0.5
    if p < 0.5:
        hidden_representations[0, :] = hidden_rep1
        hidden_representations[1, :] = hidden_rep2
    else:
        hidden_representations[0, :] = hidden_rep2
        hidden_representations[1, :] = hidden_rep1
        labels[0] = label2
        labels[1] = label1

    return hidden_representations, labels


def trial_run(dm_network, hidden_representation):
    # Run a trial: DM takes two hidden representations, previous rewards and previous actions as input and outputs an
    # action (selection of one of the two representations)
    # concatenate the two hidden reps into one input
    rep1 = hidden_representation[0]
    rep2 = hidden_representation[1]
    dm_input = np.concatenate((rep1, rep2))
    dm_input = np.array([[dm_input]])

    prediction = dm_network.predict(dm_input, batch_size=None, verbose=0, steps=None)
    choice = int(round(prediction)) # todo: change this rounding to be part of the model!

    return choice, prediction, dm_input


def check_reward(choice, labels, good_label):
    # convert choice to int if it is a float
    if isinstance(choice, float):
        choice = int(choice)

    if labels[choice] == good_label:
        reward = 1
    else:
        reward = -1

    return reward


def training(dm_network, target, prediction, choice, all_hidden_reps, optimizer, learning_rate):

    info = dm_network.fit(all_hidden_reps, target, verbose=0)
    info_history = info.history
    return dm_network, info_history


class Create_Args():
    # class for the implementation of an args data structure
    def __init__(self,nb_episodes, batch_size, consecutive_frames, training_interval, n_threads, gamma, lr, optimizer, n_timeSteps, gather_stats=0, render=0, env='empty'):
        self.nb_episode = nb_episodes
        self.batch_size = batch_size
        self.consecutive_frames = consecutive_frames
        self.training_interval = training_interval
        self.n_threads = n_threads
        self.gamma = gamma
        self.lr = lr
        self.optimizer = optimizer
        self.gather_stats = gather_stats
        self.render = render
        self.env = env
        self.n_timeSteps = n_timeSteps


def pick_digit_sample():
    hidden_reps = np.ceil(9*np.random.rand(2)) # 2 random integers between 1,9
    while hidden_reps[0] == hidden_reps[1]: # replace the second if they are equal
        hidden_reps[1] = np.floor(10*np.random.random_sample())

    labels = np.zeros([2])
    labels[0] = np.int(hidden_reps[0])
    labels[1] = np.int(hidden_reps[1])

    return hidden_reps, labels


def digit_one_hot(inp, digit_range=10):
    out = np.zeros([digit_range])
    out[np.int(inp)] = 1
    return out
