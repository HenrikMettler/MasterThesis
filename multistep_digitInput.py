import pickle
import time
import datetime

from keras.datasets import mnist
#from keras.callbacks import TensorBoard
#from helper import *
from functions import *
from Classes import *



# save_variables = 0
# # generate filename and location for storage
# currentTime = datetime.datetime.now()
# helperStr = ("decision_network%s" % currentTime)
# filename_dn = (helperStr + ".pickle")


numSets = 10

# Hyperparameters for training/testing
gamma = .9
a_size = 3 # stay, left, right
num_episode_train = 120000 # Wang: 120000
num_episode_test = 300
num_trial_per_episode = 10 # Should be 10
learning_rate = 7e-4 # Todo: Two learning rates for the critic and the actor
# optimizer = 'rmsprop' # currently rmsprop is hardcoded


# input shape
input_size = a_size*1 # left representation , 0's, right representation
empty_state = np.zeros([1, input_size]) # for fixation period
shortempty_state = np.zeros([1, input_size / a_size])

# LSTMCell parameters
num_units = 16 # Wang: 256
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

for idxSet in range(numSets):
    # empty variables for storing actions, rewards, states and trial indices
    all_actions, all_states, all_rewards, all_trialIdx, all_hidden_reps, all_good_labels, all_labels\
        = [], [], [], [], [], [], []

    # create networks
    decision_network = Network('global', a_size, num_units, input_size, gamma, learning_rate)

    time_loopstart = time.time()
    print("This is the beginning of set: ")
    print(idxSet + 1)


    # run episodes
    for idx_episode in range(num_episode_train):

        # sample 2 digits
        hidden_representations, labels = pick_digit_sample()

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
                a, log_prob = decision_network.select_action(state) # Note for Debugging: Set
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
                left_state = hidden_representations[0]
                right_state = hidden_representations[1]
                state = np.array([left_state, 0, right_state])
                state = np.array([state]) # augment to 3d as DN input


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
                            new_state = np.array([0,left_state,0])
                            new_state = np.array([new_state])
                        else:
                            new_state = np.array([0,right_state,0])
                            new_state = np.array([new_state])

                else: # The selection is done, so we want confirmation or it falls back to selecting an image
                    if a == 0:
                        if good_label == labels[rep_selected]:
                            reward = 1
                        else:
                            reward = 0
                        trial_is_finished = True
                    else:
                        reward = non_selection_reward
                        image_selection_isdone = False
                        new_state = np.array([left_state, 0, right_state])# the next state is back to initial config
                        new_state = np.array([new_state])

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

            # switch representation - action assignment with 50%
            hidden_rep1 = hidden_representations[0]
            hidden_rep2 = hidden_representations[1]
            random_float = np.random.random_sample()
            if random_float > 0.5:
                hidden_representations[0] = hidden_rep2
                hidden_representations[1] = hidden_rep1

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
        all_hidden_reps.append(hidden_representations)
        all_good_labels.append(good_label)
        all_labels.append(labels)

    time_loopend = time.time()
    print(["time passed in ",num_episode_train, " episodes:"])
    print(time_loopend - time_loopstart)

    # Test the network: #todo



    # episode_durations = np.zeros(num_episode_train)
    # for idx in range(num_episode_train):
    #     episode_durations[idx] = np.size(all_rewards[idx])



    # Saving variables

    # reduce the states to 15 indices
    # reduce_states = np.zeros([num_episode_train, num_trial_per_episode * maxtime_trial, 15])
    # for idx_episode in range(num_episode_train):
    #     num_steps_inepisode = np.size(all_states[idx_episode],0)
    #     for idx_timestep in range(num_steps_inepisode):
    #         reduce_states[idx_episode, idx_timestep, 0:4] = all_states[idx_episode][idx_timestep][0][0:4]
    #         reduce_states[idx_episode, idx_timestep, 5:9] = all_states[idx_episode][idx_timestep][0][128:132]
    #         reduce_states[idx_episode, idx_timestep, 5:9] = all_states[idx_episode][idx_timestep][0][256:260]

    # sample the episodes: every 100 episodes
    save_states = all_states[0::100,:,:]
    save_actions = all_actions[0::100]
    save_rewards = all_rewards[0::100]
    save_trialidx = all_trialIdx[0::100]
    save_labels = all_labels[0::100]
    save_goodlabels = all_good_labels[0::100]
    save_hiddenreps = all_hidden_reps[0::100]

    datastructure = Datastructure(save_states, save_actions, save_rewards, save_trialidx, save_labels, save_goodlabels, save_hiddenreps)

    helperStr = ("decisionNetwork0909_digitInputNr%s" % idxSet)
    filename = (helperStr + ".pickle")
    with open(filename, "w") as f:
        pickle.dump(datastructure, f)
    f.close()

aaa = 1