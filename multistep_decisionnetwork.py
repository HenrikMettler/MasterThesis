import matplotlib
import pickle
import json
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

# input shape
input_size = 128 # todo: automate
empty_state = np.zeros([1, input_size]) # todo: change to something else...
after_fixation_state = np.ones(np.shape(empty_state))

# Hyperparameters for training/testing
gamma = .9
a_size = 3 # stay, left, right
num_episode_train = 10 # Wang: 120000
num_episode_test = 300
num_trial_per_episode = 3 # Should be 10
learning_rate = 7e-4 # Todo: Two learning rates for the critic and the actor
# optimizer = 'rmsprop' # currently rmsprop is hardcoded

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

        # Max Time Condition & non finish condition
        while duration < maxtime_trial and not trial_is_finished:
            # Fixation Block
            while fixation_time < expected_fixation_time:
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

                if duration == maxtime_trial:
                    break

            if duration == maxtime_trial:
                break

            # change the state to after fixation state
            state = after_fixation_state


            # Selecting the Image
            while not image_selection_isdone and duration < maxtime_trial:
                #network_input = np.concatenate((state, reward, one_hot_action), 1)
                # select action
                a, log_prob = decision_network.select_action(state)
                # reset one_hot_action
                one_hot_action = np.zeros((1, a_size))
                one_hot_action[0, a] = 1

                if a == 0:
                    reward = non_selection_reward
                else:
                    reward = np.zeros((1, 1))
                    image_selection_isdone = True
                    rep_selected = np.int(a - 1)  # there is a shift of 1 because action 0 means stay
                    if a == 1: # "Left" action, pick first representation
                        state = hidden_representations[0,:,:,:]

                    elif a == 2: # "Right" action, pick second representation
                        state = hidden_representations[1,:,:,:]
                        #state = state.reshape([1, np.shape(state)[0] * np.shape(state)[1] * np.shape(state)[2]])

                    state = state.reshape([1, np.shape(state)[0] * np.shape(state)[1] * np.shape(state)[2]])

                # Append state, one_hot_action and reward
                episode_states.append(state)
                episode_actions.append(one_hot_action)
                episode_rewards.append(reward)
                episode_logprobs.append(log_prob)
                trialidx_observer.append(idx_trial)

                duration += 1
                if duration == maxtime_trial:
                    break

            if duration == maxtime_trial:
                break

            # Confirming Image Selection (by staying at image)
            # select action
            a, log_prob = decision_network.select_action(state) # Note for Debugging: Set a =0
            # Obtain Rewards (wright or wrong decision) if Image Selection is confirmed
            if a == 0:
                if good_label == labels[rep_selected]:
                    reward += 1
                else:
                    reward = 0 #no punishment for wrong selection

            episode_states.append(state)
            episode_actions.append(one_hot_action)
            episode_rewards.append(reward)
            episode_logprobs.append(log_prob)
            trialidx_observer.append(idx_trial)
            trial_is_finished = True

        # switch representation - action assignment with 50% Todo

    # convert episode_state from list into 3-dim array
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


# Test the network: #todo

time_loopend = time.time()
print(["time passed in ",num_episode_train, " episodes:"])
print(time_loopend - time_loopstart)


# save variables
if save_variables == 1:
    with open(filename_dn, "w") as f:
        pickle.dump([decision_network.policy_model, decision_network.value_model, all_actions, all_rewards, all_states, all_trialIdx], f)
    f.close()


# plotting
abc = 1

