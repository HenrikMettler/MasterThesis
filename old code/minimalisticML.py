import numpy as np
import keras.backend as K

from keras import regularizers
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Flatten, LSTM

from functions_for_minimalistic import *

num_trial_per_episode = 10
num_episode_train = 100000
num_seeds = 8
num_units_dn = 4
num_digits = 10

dataSet = generate_data(num_digits)

# build network
decision_network = build_decision_network(num_units_dn)

# train network
train_loss_matrix = np.zeros([num_episode_train, num_trial_per_episode])
for idx_episode in range(num_episode_train):

    # select data examples for this episode
    data_sample_1, data_sample_2, correct_number = select_data_for_episode(dataSet)

    for idx_trial in range(num_trial_per_episode):

        # pick one data sample
        current_sample = pick_one_sample(data_sample_1, data_sample_2)
        current_sample = np.array([current_sample])
        current_sample = np.array([current_sample])

        # feed the sample into the network and make a selection
        prediction = decision_network.predict(current_sample, batch_size=None, verbose=0, steps=None)
        choice = np.round(prediction)
        choice = choice.astype(int)
        selected_number = current_sample[choice]
        # train the network
        info = decision_network.fit()
        loss = loss_dict.get('loss', '')
        # loss matrix
        train_loss_matrix[idx_episode, idx_trial] = loss[0]


# test network

# plot training

# plot testing