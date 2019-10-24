import time
import pickle
import numpy as np

from matplotlib import pyplot as plt
from functions import *
from Classes import *
from A2C.a2c import A2C

from datetime import date

""" Import Data, Set Parameters"""

# debug_mode = False
#input_mode = 'minimal'  # options: minimal


# Hyperparameters for training/testing
num_steps_unrolled = 8
gamma = .9
gammas = [.9] #[0.5, 0.8, 0.9, 0.95, 0.99]
a_size = 2  # left, right
num_episode_train = 70000 # Wang: 120000
num_trial_per_episode = 10  # Should be 10
learning_rate = 7e-4 #
learning_rates = [7e-4] #[1e-4, 2e-4, 5e-4, 1e-3]
num_lstm_units = 16  # Wang: 256 num_lstm_units_array = [16, 48, 256]


# input shape
input_size = 3 * 10 # 10 is the dimensionality one hot digit encoding

# environment parameters
reward = 1
false_reward = 0  # *np.ones((1,1))
#target_fixation_time = 2

# Configuration parameters
save_variables = 0
save_figures = 0

hidden_representations = np.zeros([2,10])
""" RUN """
for learning_rate in learning_rates:

    # define empty states as initial values
    empty_state_memory = np.zeros([num_steps_unrolled, input_size])  # for fixation period
    shortempty_state = np.zeros(10)

    # create instance of Advantage Actor Critic
    algorithm = A2C(a_size, input_size, num_lstm_units, num_steps_unrolled, gamma, learning_rate, print_summary=True)

    # empty variables for storing actions, rewards, states and trial indices
    reward_pertrial_matrix = np.zeros([num_episode_train, num_trial_per_episode])
    rewarded_image = np.zeros([num_episode_train, num_trial_per_episode])
    selected_image = np.zeros([num_episode_train, num_trial_per_episode])
    discounted_rewards = np.zeros([num_episode_train, num_trial_per_episode])
    state_values = []
    action_values = np.zeros([num_episode_train, num_trial_per_episode, a_size])
    run_duration = 0
    model = []

    time_start = time.time()
    state_memory = np.array(empty_state_memory)

    # run episodes
    for idx_episode in range(num_episode_train):

        # Load 2 hidden representations and the corresponding labels
        digit_samples, labels = pick_digit_sample()
        hidden_representations[0,:] = digit_one_hot(digit_samples[0])
        hidden_representations[1,:] = digit_one_hot(digit_samples[1])

        # pick one of the representations to be the rewarded one
        q = np.random.random_sample()
        if q > 0.5:
            good_label = labels[1]
        else:
            good_label = labels[0]

        # Reset parameters at beginning of an episode
        state_values_episode = []
        episode_states, episode_actions, episode_actions_scalar, episode_rewards, trialidx_observer = [], [], [], [], []

        for idx_trial in range(num_trial_per_episode):

            # store which image is rewarded
            if good_label == labels[0]:
                rewarded_image[idx_episode, idx_trial] = 0
            else:
                rewarded_image[idx_episode, idx_trial] = 1

            # create the new state
            left_state = np.squeeze(hidden_representations[0,:])
            right_state = np.squeeze(hidden_representations[1,:])
            new_state = np.concatenate([left_state, shortempty_state, right_state], 0)

            # replace the latest state in the memory with the current state
            last_states = np.array([state_memory[1:, :]])
            state_memory[0:num_steps_unrolled - 1, :] = last_states
            state_memory[num_steps_unrolled - 1, :] = new_state
            # pick an action & observe state values
            a, action_probs = algorithm.select_action(state_memory)# a = 0 or 1
            action_values[idx_episode, idx_trial, :] = action_probs
            one_hot_action = np.zeros((1, a_size))
            one_hot_action[0, a] = 1

            current_state_value = algorithm.critic.predict(state_memory)
            # observe reward
            rep_selected = a
            if good_label == labels[rep_selected]:
                reward = 1
            else:
                reward = false_reward

            # Append state, one_hot_action, reward
            episode_states.append(np.array(state_memory))
            episode_actions.append(one_hot_action)
            # episode_actions_scalar.append(np.int(a))
            episode_rewards.append(np.float(reward))
            # trialidx_observer.append(idx_trial)


            # Todo: create a switch to turn it on an off
            # switch representation - action assignment with 50%
            hidden_representations, labels = mix_up(left_state, right_state, labels)

            reward_pertrial_matrix[idx_episode, idx_trial] = reward
            selected_image[idx_episode, idx_trial] = a
            state_values_episode.append(current_state_value)

        # append all the critic values of the episode
        state_values.append(state_values_episode)

        # convert from lists into 3-dim array
        episode_states = np.array(episode_states)
        episode_actions = np.array(episode_actions)
        episode_rewards = np.array(episode_rewards)

        # train the algorithm using the state, action and reward memory
        #discounted_rewards[idx_episode, :] =
        algorithm.train_models(episode_states, episode_actions, episode_rewards, num_steps_unrolled)


    time_loopend = time.time()
    print(["time passed in ", num_episode_train, " episodes:"])
    print(time_loopend - time_start)

    """ Variable Saving and Plotting """
    run_duration = time_loopend - time_start
    today = date.today()
    month = today.month
    day = today.day
    filename = '{0}{1}_nEpisode:{2}_nUnroll:{3}_gamma:{4}_lr:{5}_nLstm:{6}WoFixation.pickle'.format(month, day, num_episode_train, num_steps_unrolled, gamma,
                                                                             learning_rate, num_lstm_units)
    with open('data/{0}'.format(filename), "w") as f:
        pickle.dump(
            [reward_pertrial_matrix, selected_image, rewarded_image, state_values, action_values, run_duration,
             false_reward], f)
    f.close()

