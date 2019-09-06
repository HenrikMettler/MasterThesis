import matplotlib
import pickle
import time
import datetime

from keras.datasets import mnist
#from keras.callbacks import TensorBoard
#from helper import *
from functions import *
from Classes import *

matplotlib.use("TkAgg") # matplotlib import adapted for use on MacOS

from matplotlib import pyplot as plt

save_variables = 0
# generate filename and location for storage
currentTime = datetime.datetime.now()
helperStr = ("decision_network%s" % currentTime)
filename_dn = (helperStr + ".pickle")


# load the autoencoder model
filename_autoencoder = "autoencoder_mnist2019-07-11 12:34:09.098789.pickle" # reference to the autoencoder file

infile = open(filename_autoencoder, 'rb')
autoencoderModel, encoderModel = pickle.load(infile)
infile.close()

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


#after_fixation_state = np.ones(np.shape(empty_state))

# Hyperparameters for training/testing
gamma = .9
a_size = 3 # stay, left, right
num_episode_train = 1000 # Wang: 120000
num_episode_test = 300
num_trial_per_episode = 10 # Should be 10
learning_rate = 7e-4 # Todo: Two learning rates for the critic and the actor
# optimizer = 'rmsprop' # currently rmsprop is hardcoded


# input shape
input_size = a_size*128 # left representation , 0's, right representation todo: automate
empty_state = np.zeros([1, input_size]) # for fixation period todo: change to something else...
shortempty_state = np.zeros([1, input_size / a_size])

# LSTMCell parameters
num_units = 48 # Wang: 256
activation = 'tanh'
recurrent_activation = 'hard_sigmoid'
use_bias = True

# environment parameters
maxtime_trial = 20 # todo what is a meaningful value?
expected_fixation_time = 2
non_fixation_reward = - 0.01*np.ones((1,1))
fixation_reward = 0.2*np.ones((1,1))
non_selection_reward = - 0.01*np.ones((1,1))
target_fixation_time = 2


# empty variables for storing actions, rewards, states and trial indices
all_actions, all_states, all_rewards, all_trialIdx = [], [], [], []

# create networks
decision_network = Network('global', a_size, num_units, input_size, gamma, learning_rate)

time_loopstart = time.time()

# run episodes
for idx_episode in range(num_episode_train):

    print("This is the beginning of episode: ")
    print(idx_episode+1)

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

    # perform the trials
    for idx_trial in range(num_trial_per_episode):

        # randomly assign left / right actions to the representations
        # assign reward to correct side
        duration = 0
        reward = np.zeros((1, 1))
        one_hot_action = np.zeros((1, a_size))
        fixation_time = 0 # counts the fixation time
        image_selection_isdone = False
        trial_is_finished = False
        state = empty_state # change to representation of full black image

        # Fixation Block
        while fixation_time < expected_fixation_time and duration < maxtime_trial:
            # select action
            a, log_prob = decision_network.select_action(state) # Note for Debugging: Set a = 0
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
            episode_logprobs.append(log_prob)
            trialidx_observer.append(idx_trial)
            duration += 1

        # Image Selection And Confirmation Block
        if fixation_time == expected_fixation_time:
            # change the state to Left_rep, 0's, Right_rep
            left_state = hidden_representations[0, :, :, :]  # Todo: Move the reshaping somewhere else
            left_state = left_state.reshape([1, np.shape(left_state)[0] * np.shape(left_state)[1] * np.shape(left_state)[2]])
            right_state = hidden_representations[1, :, :, :]
            right_state = right_state.reshape([1, np.shape(right_state)[0] * np.shape(right_state)[1] * np.shape(right_state)[2]])

            state = np.concatenate([left_state, shortempty_state, right_state],1)

        while duration < maxtime_trial and not trial_is_finished:

            # select action
            a, log_prob = decision_network.select_action(state)
            # reset one_hot_action
            one_hot_action = np.zeros((1, a_size))
            one_hot_action[0, a] = 1
            if not image_selection_isdone:
                if a == 0:
                    reward = non_selection_reward
                    new_state = state # state remains the same for next trial
                else:
                    reward = np.zeros((1, 1))
                    image_selection_isdone = True
                    rep_selected = np.int(a-1) # Shift by -1 because action 1 stands for 0 and action 2 for 1
                    if a == 1:
                        new_state = np.concatenate([shortempty_state, left_state, shortempty_state],1)
                    else:
                        new_state = np.concatenate([shortempty_state, right_state, shortempty_state],1)
            else: # The selection is done, so we want confirmation or it falls back
                if a == 0:
                    if good_label == labels[rep_selected]:
                        reward = 1
                    else:
                        reward = 0
                    trial_is_finished = True
                else:
                    reward = non_selection_reward
                    image_selection_isdone = False
                    new_state = np.concatenate([left_state, shortempty_state, right_state],1) # the next state is back to initial config

            # Append state, one_hot_action and reward
            episode_states.append(state)
            episode_actions.append(one_hot_action)
            episode_rewards.append(reward)
            episode_logprobs.append(log_prob)
            trialidx_observer.append(idx_trial)
            duration += 1

            # update state (updates needs to be done after attaching the state to episode states
            if not trial_is_finished: # this computation is meaningless if the trial is finished
                state = new_state

        # switch representation - action assignment with 50% Todo: make nicer
        hidden_rep1 = hidden_representations[0, :, :, :]
        hidden_rep2 = hidden_representations[1, :, :, :]
        hidden_representations, labels = mix_up(hidden_rep1, hidden_rep2, labels)

    # convert from lists into 3-dim array
    episode_states_asarray = np.array(episode_states)
    episode_actions_asarray = np.array(episode_actions)
    episode_rewards_asarray = np.array(episode_rewards)
    trialidx_observer_asarray = np.array(trialidx_observer)

    # Train Network
    decision_network.train(episode_states_asarray, episode_actions_asarray, episode_rewards_asarray, episode_logprobs)

    # append all the states, actions, rewards and trial indices to the over all data
    all_actions.append(episode_actions_asarray)
    all_states.append(episode_states_asarray)
    all_rewards.append(episode_rewards)
    all_trialIdx.append(trialidx_observer_asarray)


time_loopend = time.time()
print(["time passed in ",num_episode_train, " episodes:"])
print(time_loopend - time_loopstart)

# Test the network: #todo

latest_rewards = all_rewards[num_episode_train-100:]
latest_trial_idx = all_trialIdx[num_episode_train-100:]

episode_durations = np.zeros(num_episode_train)
for idx in range(num_episode_train):
    episode_durations[idx] = np.size(all_rewards[idx])