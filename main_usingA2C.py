import tensorflow as tf
import time
import matplotlib
import pickle
import numpy as np

from matplotlib import pyplot as plt

from keras.datasets import mnist
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical

from functions import *
from Classes import *
from A2C.a2c import A2C

debug_mode = False

filename = "autoencoder_mnist2019-07-11 12:34:09.098789.pickle"

infile = open(filename,'rb')
autoencoderModel, encoderModel = pickle.load(infile)

# Import data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Prepare Input: Normalizing, Flatten Images
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# Hyperparameters for training/testing
gamma = .9
a_size = 3 # stay, left, right
num_episode_train = 60000 # Wang: 120000
num_episode_test = 300
num_trial_per_episode = 7 # Should be 10
learning_rate = 7e-4 # Todo: Two learning rates for the critic and the actor
# optimizer = 'rmsprop' # currently rmsprop is hardcoded

# input shape
input_size = a_size*128 # left representation , 0's, right representation todo: automate
empty_state = np.zeros([1, input_size]) # for fixation period todo: change to something else...
shortempty_state = np.zeros([1, input_size / a_size])

# LSTMCell parameters
num_lstm_units = 48 # Wang: 256
num_dense_units = 48
activation = 'tanh'
recurrent_activation = 'hard_sigmoid'
use_bias = True

# environment parameters
maxtime_trial = 20
expected_fixation_time = 2
non_fixation_reward = - 0.01*np.ones((1,1))
fixation_reward = 0.2*np.ones((1,1))
non_selection_reward = - 0.01*np.ones((1,1))
target_fixation_time = 2

# create Algorithm

algorithm = A2C(a_size, input_size, num_lstm_units, num_dense_units, gamma, learning_rate, print_summary=True)


# empty variables for storing actions, rewards, states and trial indices
all_actions, all_states, all_rewards, all_trialIdx, all_hidden_reps, all_good_labels, all_labels\
    = [], [], [], [], [], [], []

time_loopstart = time.time()
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
    episode_states, episode_actions, episode_rewards, episode_logprobs, trialidx_observer = [], [], [], [], []

    for idx_trial in range(num_trial_per_episode):

        # initialize some variables for each trial
        duration, reward,  one_hot_action = 0, np.zeros((1, 1)), np.zeros((1, a_size))
        fixation_time = 0  # counts the fixation time
        fixation_isdone, image_selection_isdone, trial_is_finished  = False, False, False
        state = empty_state  # change to representation of full black image

        # Fixation Block
        while fixation_time < expected_fixation_time and duration < maxtime_trial:
            # select action
            a = algorithm.select_action(state)  # Note for Debugging: Set a = 0
            # reset one_hot_action
            one_hot_action = np.zeros((1, a_size))
            one_hot_action[0, a] = 1

            if a == 0:
                fixation_time += 1
                # Collect Fixation Reward if success
                if fixation_time == expected_fixation_time:
                    reward = fixation_reward
                else:
                    reward = 0
            else:
                fixation_time = 0
                reward = non_fixation_reward

            # Append state, one_hot_action, reward and log probs
            episode_states.append(state)
            episode_actions.append(one_hot_action)
            episode_rewards.append(reward)
            trialidx_observer.append(idx_trial)
            duration += 1

        # Image Selection And Confirmation Block
        if fixation_time == expected_fixation_time: # update state
            # change the state to Left_rep, 0's, Right_rep
                left_state = hidden_representations[0, :]
                left_state = np.asarray([left_state])
                right_state = hidden_representations[1, :]
                right_state = np.asarray([right_state])
                state = np.concatenate([left_state, shortempty_state, right_state], 1)

        while duration < maxtime_trial and not trial_is_finished:

            # select action
            a = algorithm.select_action(state)
            # reset one_hot_action
            one_hot_action = np.zeros((1, a_size))
            one_hot_action[0, a] = 1
            if not image_selection_isdone:
                if a == 0:
                    reward = non_selection_reward
                    new_state = state  # state remains the same for next trial
                else:
                    reward = np.zeros((1, 1))
                    image_selection_isdone = True
                    rep_selected = np.int(a - 1)  # Shift by -1 because action 1 stands for 0 and action 2 for 1
                    if a == 1:
                        new_state = np.concatenate([shortempty_state, left_state, shortempty_state], 1)
                    else:
                        new_state = np.concatenate([shortempty_state, right_state, shortempty_state], 1)
            else:  # The selection is done, so we want confirmation or it falls back to selecting an image
                if a == 0:
                    if good_label == labels[rep_selected]:
                        reward = 1
                    else:
                        reward = 0
                    trial_is_finished = True
                else:
                    reward = non_selection_reward
                    image_selection_isdone = False
                    new_state = np.concatenate([left_state, shortempty_state, right_state],
                                               1)  # the next state is back to initial config

            # Append state, one_hot_action and reward
            episode_states.append(state)
            episode_actions.append(one_hot_action)
            episode_rewards.append(reward)
            trialidx_observer.append(idx_trial)
            duration += 1

            # update state (updates needs to be done after attaching the state to episode states
            if not trial_is_finished:  # this computation is meaningless if the trial is finished
                state = new_state

            # Todo: create a switch to turn it on an off
            # switch representation - action assignment with 50%
            hidden_rep1 = hidden_representations[0, :]
            hidden_rep2 = hidden_representations[1, :]
            hidden_representations, labels = mix_up(hidden_rep1, hidden_rep2, labels)

    # convert from lists into 3-dim array
    episode_states_asarray = np.array(episode_states)
    episode_actions_asarray = np.array(episode_actions)
    episode_rewards_asarray = np.array(episode_rewards)
    trialidx_observer_asarray = np.array(trialidx_observer)

    if debug_mode == True:
        filename = 'sRa_forDebug.pickle'
        infile = open(filename, 'rb')
        episode_states_asarray, episode_actions_asarray, episode_rewards_asarray = pickle.load(infile)

    # train the network with data from the episode
    algorithm.train_models(episode_states_asarray, episode_actions_asarray, episode_rewards_asarray)

    # append all the states, actions, rewards and trial indices to the over all data
    all_actions.append(episode_actions_asarray)
    all_states.append(episode_states_asarray)
    all_rewards.append(episode_rewards)
    all_trialIdx.append(trialidx_observer_asarray)
    all_hidden_reps.append(hidden_representations)
    all_good_labels.append(good_label)
    all_labels.append(labels)

time_loopend = time.time()
print(["time passed in ",num_episode_train, " episodes:"])
print(time_loopend - time_loopstart)

a = 1
# # save some variable for train debugging
# filename = 'sRa_forDebug.pickle'
# with open(filename, "w") as f:  # The w stands for write
#     pickle.dump([episode_states_asarray, episode_actions_asarray, episode_rewards_asarray], f)
# f.close()
# filename = 'sRa_forDebug.pickle'
# infile = open(filename,'rb')
# episode_states_asarray, episode_actions_asarray, episode_rewards_asarray = pickle.load(infile)

# save rewards and trial idx
filename = 'reward_trialIdx1309.pickle'
with open(filename, "w") as f:
    pickle.dump([all_rewards, all_trialIdx], f)
f.close()

filename = 'reward_trialIdx1309.pickle'
infile = open(filename,'rb')
all_rewards, all_trialIdx = pickle.load(infile)

# prepare data for plotting
reward_pertrial_matrix = np.zeros([num_episode_train, num_trial_per_episode])
for idx_episode in range(num_trial_per_episode):
    current_rewards = all_rewards[idx_episode]
    current_trial_inidices = all_trialIdx[idx_episode]
    num_timesteps = len(current_rewards)
    for idx_timestep in range(num_timesteps):
        current_trial_idx = current_trial_inidices[idx_timestep]
        reward_pertrial_matrix[idx_episode, current_trial_idx] += current_rewards[idx_timestep]

plt.figure()
for idx in range (num_trial_per_episode):
    plt.subplot(2,4,idx+1)
    plt.plot(reward_pertrial_matrix[:,idx])
    if idx == 0:
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward per Trial, Trial %s', idx+1)
    else:
        plt.title('Trial %s', idx+1)
    plt.xlim([-100, 61000])
    plt.ylim([-0.4, 1.5])
    plt.grid(True)
    plt.show()