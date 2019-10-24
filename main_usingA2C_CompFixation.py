import time
import pickle
import numpy as np

from matplotlib import pyplot as plt
from keras.datasets import mnist
from functions import *
from Classes import *
from A2C.a2c import A2C

from datetime import date

""" Import Data, Set Parameters"""

# debug_mode = False
input_mode = 'minimal'  # options: minimal (autoencoder doesn't work in this file...)


num_sets = 10

# Hyperparameters for training/testing
num_steps_unrolled = 20
#num_steps_unrolled_array = [10, 20, 50, 100]
gamma = .9 # discount factor
#gammas = [0.5, 0.8, 0.9, 0.95, 0.99]
a_size = 3  # stay, left, right
num_episode_train = 20000 # Wang: 120000
num_episode_test = 300
num_trial_per_episode = 10  # Should be 10
learning_rate = 7e-4
learning_rates = [5e-4, 7e-4, 1e-3, 2e-3]
num_lstm_units = 16  # Wang: 256
num_lstm_units_array = [16, 48, 256]

# LSTM parameters + Optimizer (currently hardcoded)
# activation = 'tanh'
# recurrent_activation = 'hard_sigmoid'
# use_bias = True
# optimizer = 'rmsprop' # currently rmsprop is hardcoded

# input shape
if input_mode == 'minimal':
    input_size = a_size * 10 # 10 is the size of the one hot digit encoding
else:
    raise NameError("Input mode is not available, pick 'minimal '")

# environment parameters
maxtime_trial = 50
expected_fixation_time = 3
non_fixation_reward = - 0.01  # *np.ones((1,1))
fixation_reward = 0.2  # *np.ones((1,1))
expected_selection_time = 2
non_selection_reward = -0.01  # *np.ones((1,1))
wrong_selection_reward = -1  # *np.ones((1,1))
target_fixation_time = 2

# Configuration parameters
save_variables = 1
save_figures = 0

""" RUN """

# define empty states as initial values
empty_state_memory = np.zeros([num_steps_unrolled, input_size])  # for fixation period
shortempty_state = np.zeros([1, input_size / a_size])
fixation_vector = np.asarray([[0,1,0,1,0,1,0,1,0,1]])
fixation_state = np.concatenate([shortempty_state, fixation_vector, shortempty_state],1)
empty_state_memory[num_steps_unrolled-1, :] = fixation_state


# create instance of Advantage Actor Critic
algorithm = A2C(a_size, input_size, num_lstm_units, num_steps_unrolled, gamma, learning_rate, print_summary=True)

# empty variables for storing actions, rewards, states and trial indices
reward_pertrial_matrix = np.zeros([num_episode_train, num_trial_per_episode])
trial_durations = np.zeros([num_episode_train, num_trial_per_episode])
rewarded_image = np.zeros([num_episode_train, num_trial_per_episode])
selected_image = np.zeros([num_episode_train, num_trial_per_episode])
state_values = []
run_duration = 0
model = []
# all_actions, all_states, all_rewards, all_trialIdx, all_hidden_reps, all_good_labels, all_labels\
#     = [], [], [], [], [], [], []

time_start = time.time()

for idx_set in range(4, num_sets):
    # run episodes
    for idx_episode in range(num_episode_train):

        # Load 2 hidden representations and the corresponding labels
        if input_mode == 'minimal':
            # Load 2 hidden representations and the corresponding labels
            digit_samples, labels = pick_digit_sample()
            hidden_representations = np.zeros([2, 10])
            hidden_representations[0, :] = digit_one_hot(digit_samples[0])
            hidden_representations[1, :] = digit_one_hot(digit_samples[1])

        # pick one of the representations to be the rewarded one
        q = np.random.random_sample()
        if q > 0.5:
            good_label = labels[1]
            rewarded_image[idx_episode] = 2
        else:
            good_label = labels[0]
            rewarded_image[idx_episode] = 1

        # Reset parameters at beginning of an episode
        state_values_episode = []
        episode_states, episode_actions, episode_actions_scalar, episode_rewards, trialidx_observer = [], [], [], [], []
        state_memory = np.array(empty_state_memory)

        for idx_trial in range(num_trial_per_episode):

            # Reset trial variable dependant variables
            duration, reward, one_hot_action = 0, 0, np.zeros((1, a_size))
            fixation_time, selection_time, last_action = 0, 0, 0  # counts the fixation and selection time
            fixation_isdone, image_selection_isdone, trial_is_finished = False, False, False
            trial_reward = 0
            second_2last_action = 0
            state_values_trial = []

            # Fixation Block
            while fixation_time < expected_fixation_time and duration < maxtime_trial:
                # select action
                a, action_probs = algorithm.select_action(state_memory)  # Note for Debugging: Set a = 0
                current_state_value = algorithm.critic.predict(state_memory)
                state_values_trial.append(current_state_value)
                # reset one_hot_action
                one_hot_action = np.zeros((1, a_size))
                one_hot_action[0, a] = 1

                if a == 0:
                    fixation_time += 1
                    # Collect Fixation Reward if success
                    if fixation_time == expected_fixation_time:
                        reward = fixation_reward
                        # change the state to Left_rep, 0's, Right_rep
                        if input_mode == 'autoencoder':
                            left_state = hidden_representations[0, :]
                            left_state = np.asarray([left_state])
                            right_state = hidden_representations[1, :]
                            right_state = np.asarray([right_state])
                            new_state = np.concatenate([left_state, shortempty_state, right_state], 1)
                        elif input_mode == 'minimal':
                            left_state = np.asarray([hidden_representations[0, :]])
                            right_state = np.asarray([hidden_representations[1, :]])
                            new_state = np.concatenate([left_state, shortempty_state, right_state], 1)

                    else:
                        reward = 0
                        new_state = np.array([fixation_state])
                else:
                    fixation_time = 0
                    reward = non_fixation_reward
                    new_state = np.array([fixation_state])

                # Append state, one_hot_action, reward and log probs
                episode_states.append(np.array(state_memory))
                episode_actions.append(one_hot_action)
                # episode_actions_scalar.append(np.int(a))
                episode_rewards.append(np.float(reward))
                # trialidx_observer.append(idx_trial)
                trial_reward += reward
                duration += 1

                # reconfigure the state memory:
                last_states = np.array([state_memory[1:, :]])
                state_memory[0:num_steps_unrolled - 1, :] = last_states
                state_memory[num_steps_unrolled - 1, :] = new_state

            while duration < maxtime_trial and not trial_is_finished:

                # select action
                a, action_probs = algorithm.select_action(state_memory) # a = 1, a = 0
                if last_action == 0: # the last action needs to be reset if it hasn't been called yet
                    last_action = a
                current_state_value = algorithm.critic.predict(state_memory)
                state_values_trial.append(current_state_value)
                # reset one_hot_action
                one_hot_action = np.zeros((1, a_size))
                one_hot_action[0, a] = 1
                if not image_selection_isdone:
                    if a == 0 or a != last_action:
                        reward = non_selection_reward
                        new_state = new_state  # state remains the same for next trial
                        selection_time = 0
                    else:
                        reward = np.zeros((1, 1))
                        if selection_time == expected_selection_time:
                            image_selection_isdone = True
                            rep_selected = np.int(a - 1)  # Shift by -1 because action 1 stands for 0 and action 2 for 1
                            if a == 1:
                                new_state = np.concatenate([shortempty_state, left_state, shortempty_state], 1)
                                second_2last_action = 1
                            else:
                                new_state = np.concatenate([shortempty_state, right_state, shortempty_state], 1)
                                second_2last_action = 2
                        else:
                            selection_time += 1
                            new_state = new_state # state remains the same for next trial
                else:  # The selection is done, so we want confirmation or it falls back to selecting an image
                    if a == 0:
                        if good_label == labels[rep_selected]:
                            reward = 1
                        else:
                            reward = wrong_selection_reward
                        trial_is_finished = True
                        new_state = np.array(fixation_state)
                    else:
                        reward = non_selection_reward
                        image_selection_isdone = False
                        new_state = np.concatenate([left_state, shortempty_state, right_state],
                                                       1)  # the next state is back to initial config

                # Append state, one_hot_action reward and trial idx
                episode_states.append(np.array(state_memory))
                episode_actions.append(one_hot_action)
                episode_rewards.append(np.float(reward))
                trial_reward += reward
                duration += 1
                last_action = a

                # reconfigure the state memory:
                last_states = np.array(state_memory[1:, :])
                state_memory[0:num_steps_unrolled - 1, :] = last_states
                state_memory[num_steps_unrolled - 1, :] = new_state

                # Todo: create a switch to turn it on an off
                # switch representation - action assignment with 50%
                # hidden_rep1 = hidden_representations[0, :]
                # hidden_rep2 = hidden_representations[1, :]
                # hidden_representations, labels = mix_up(hidden_rep1, hidden_rep2, labels)

            reward_pertrial_matrix[idx_episode, idx_trial] = trial_reward
            trial_durations[idx_episode, idx_trial] = duration
            selected_image[idx_episode, idx_trial] = second_2last_action
            state_values_episode.append(state_values_trial)

        # append all the critic values of the episode
        state_values.append(state_values_episode)

        # convert from lists into 3-dim array
        episode_states = np.array(episode_states)
        episode_actions = np.array(episode_actions)
        episode_rewards = np.array(episode_rewards)



        # train the network with data from the episode
        algorithm.train_models(episode_states, episode_actions, episode_rewards, num_steps_unrolled)

    time_loopend = time.time()
    print(["time passed in ", num_episode_train, " episodes:"])
    print(time_loopend - time_start)

    """ Variable Saving and Plotting """
    run_duration = time_loopend - time_start
    today = date.today()
    month = today.month
    day = today.day
    filename = '{0}{1}_nEpisode:{2}_nUnroll:{3}_gamma:{4}_lr:{5}_nLstm:{6}_set:{7}.pickle'.format(month, day, num_episode_train,
                                                                                              num_steps_unrolled, gamma,
                                                                                              learning_rate, num_lstm_units, idx_set)
    with open('data/{0}'.format(filename), "w") as f:
        pickle.dump(
                [reward_pertrial_matrix, trial_durations, selected_image, rewarded_image, state_values, run_duration,
                 non_selection_reward, non_fixation_reward, wrong_selection_reward], f)
    f.close()

