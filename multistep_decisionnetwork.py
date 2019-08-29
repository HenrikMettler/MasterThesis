import matplotlib
import pickle
import datetime
#import tensorflow_probability as tfp
import time

from keras.models import Model
from keras import regularizers
from keras.datasets import mnist
from keras import backend as K
from keras.callbacks import TensorBoard
from helper import *
from functions import *
from Classes import *

#from AdvantageActorCritic import *
from Classes import *
matplotlib.use("TkAgg") # matplotlib import adapted for use on MacOS

from matplotlib import pyplot as plt

# filename = "autoencoder_mnist2019-07-05 12:00:03.686985.pickle"
filename= "autoencoder_mnist2019-07-11 12:34:09.098789.pickle"

infile = open(filename,'rb')
autoencoderModel, encoderModel = pickle.load(infile)
#infile.close()

# Autoencoder, Encoder model summary
print("This is the Autoencoder model summary")
autoencoderModel.summary()
print("This is the Encoder model summary")
encoderModel.summary()

# Import data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Prepare Input: Normalizing, Flatten Images
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# input shape
input_size = 128 # todo: automate
empty_state = np.zeros([1, input_size])
# Hyperparameters for training/testing
gamma = .9
a_size = 3 # stay, left, right
num_episode_train = 120000
num_episode_test = 300
num_trial_per_episode = 10
learning_rate = 7e-4
optimizer = 'rmsprop'
loss = 'mean_squared_error'

# LSTMCell parameters
num_units = 48 # Wang: 256
activation = 'tanh'
recurrent_activation = 'hard_sigmoid'
use_bias = True

# environment parameters
maxtime_trial = 20 # todo what is meaningful value?
expected_fixation_time = 2
non_fixation_reward = - 0.01
fixation_reward = 0.2
non_selection_reward = - 0.01
target_fixation_time = 2

# create networks
decision_network = Network('global', a_size, num_units, input_size, gamma, learning_rate, optimizer)

# run episodes
for idx_episode in range(num_episode_train):

    # sample hidden representations
    hidden_representations, labels \
        = create_sample_representation(encoderModel, x_train, y_train, is_training=1)

    # pick one of the actions to be the rewarded one
    q = np.random.random_sample()
    if q > 0.5:
        good_label = labels[1]
    else:
        good_label = labels[0]

    # Set some parameters to episode start
    episode_action, episode_reward = [], []
    reward, t = np.zeros((1,1)), np.zeros((1,1))
    one_hot_action = np.zeros((1, a_size))

    # perform the trials
    for idx_trial in range(num_trial_per_episode):

        # randomly assign left / right actions to the representations
        # assign reward to correct side
        time = 0
        # location = 0 # indicates where the agent is at the moment
        fixation_time = 0 # counts the fixation time
        image_selection_isdone = False
        trial_is_finished = False
        state = empty_state

        # Max Time Condition
        while time < maxtime_trial and not trial_is_finished:
            # Fixation Block
            while fixation_time < expected_fixation_time:
                # create network input (state, r(t-1), one_hot_action)
                network_input = np.concatenate((state, reward, one_hot_action), 1)
                # select action
                a = decision_network.select_action(network_input) # Note for Debugging: Set a =0
                # reset one_hot_action
                one_hot_action = np.zeros((1, a_size))
                one_hot_action[0, a] = 1

                if a == 0:
                    fixation_time += 1
                else:
                    fixation_time = 0
                    reward += non_fixation_reward
                time += 1

            # Collect Fixation Reward
            if fixation_time == expected_fixation_time:
                reward += fixation_reward

            # Selecting the Image
            while not image_selection_isdone:
                network_input = np.concatenate((state, reward, one_hot_action),1)
                # select action
                a = decision_network.select_action(network_input)
                # reset one_hot_action
                one_hot_action = np.zeros((1, a_size)) # Note for Debugging: Set a !=0
                one_hot_action[0, a] = 1

                if a == 0:
                    reward += non_selection_reward
                else:
                    image_selection_isdone = True
                    rep_selected = a - 1  # there is a shift of 1 because action 0 means stay
                    if a == 1: # "Left" action, pick first representation
                        state = hidden_representations[0,:,:,:]
                        state = state.reshape([1,np.shape(state)[0] * np.shape(state)[1] * np.shape(state)[2]])

                    elif a == 2: # "Right" action, pick second representation
                        state = hidden_representations[1,:,:,:]
                        state = state.reshape([1, np.shape(state)[0] * np.shape(state)[1] * np.shape(state)[2]])

                time += 1

            # Confirming Image Selection (by sta
            network_input = np.concatenate((state, reward, one_hot_action),1)
            # select action
            a = decision_network.select_action(network_input) # Note for Debugging: Set a =0
            # Obtain Rewards (wright or wrong decision) if Image Selection is confirmed
            if a == 0:
                if good_label == labels[rep_selected]:
                    reward += 1
        # Train Network Todo: Confirm it is here!
        decision_network.train()


        # switch representation - action assignment with 50%

