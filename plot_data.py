import pickle
import numpy as np

from matplotlib import pyplot as plt


filename1 = 'data/930_nEpisode:20000_nUnroll:20_gamma:0.9_lr:0.0007_nLstm:16_set:0.pickle'
filename2 = 'data/101_nEpisode:20000_nUnroll:20_gamma:0.9_lr:0.0007_nLstm:16_set:1.pickle'
filename3 = 'data/101_nEpisode:20000_nUnroll:20_gamma:0.9_lr:0.0007_nLstm:16_set:2.pickle'
filename4 = 'data/101_nEpisode:20000_nUnroll:20_gamma:0.9_lr:0.0007_nLstm:16_set:3.pickle'

infile = open(filename1, "rb")
reward_pertrial_matrix_1, trial_durations_1, selected_image_1, rewarded_image_1, state_values_1, run_duration_1, \
non_selection_reward_1, non_fixation_reward_1, wrong_selection_reward_1 = pickle.load(infile)

infile = open(filename2, "rb")
reward_pertrial_matrix_2, trial_durations_2, selected_image_2, rewarded_image_2, state_values_2, run_duration_2, \
non_selection_reward_2, non_fixation_reward_2, wrong_selection_reward_2 = pickle.load(infile)

infile = open(filename3, "rb")
reward_pertrial_matrix_3, trial_durations_3, selected_image_3, rewarded_image_3, state_values_3, run_duration_3, \
non_selection_reward_3, non_fixation_reward_3, wrong_selection_reward_3 = pickle.load(infile)

infile = open(filename4, "rb")
reward_pertrial_matrix_4, trial_durations_4, selected_image_4, rewarded_image_4, state_values_4, run_duration_4, \
non_selection_reward_4, non_fixation_reward_4, wrong_selection_reward_4 = pickle.load(infile)





more_val_action = np.max(action_values,2)

reward_per_Episode = np.sum(reward_pertrial_matrix,1)

# plt.figure(1)
# #plt.subplot(1,2,1)
# plt.plot(more_val_action[:, 0])
#
# plt.xlabel('Episode')
# plt.ylabel('Max Action probability Trial 1')
# plt.title('Action probabilities ' )
#
# plt.xlim([-10, np.size(rewarded_image,0)*1.1])
# plt.ylim([-0.4, 1.5])
# plt.grid(True)
# plt.show()




plt.figure(2)
for idx in range(10):
    plt.subplot(2, 5, idx + 1)
    plt.plot(action_values[:, idx, 0])

    plt.xlabel('Episode')
    plt.ylabel('Action probability Trial 1 - Action 0')
    plt.title('Action probabilities ' )

    plt.xlim([-10, np.size(rewarded_image,0)*1.1])
    plt.ylim([-0.4, 1.5])
    plt.grid(True)
    plt.show()


plt.figure(3)
plt.plot(reward_per_Episode)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Summed Reward per Episode')
plt.ylim([-1,11])
plt.xlim([-10, np.size(rewarded_image,0)*1.1])

smoothing_coeff = 0.95
smooth_reward_per_episode = np.zeros(np.size(rewarded_image,0))
smooth_reward_per_episode[0] = reward_per_Episode[0]
for idx_episode in range(1, np.size(rewarded_image,0)):
    smooth_reward_per_episode[idx_episode] = smoothing_coeff * smooth_reward_per_episode[idx_episode-1] + \
                                             (1-smoothing_coeff) * reward_per_Episode[idx_episode]

plt.figure(4)
plt.plot(smooth_reward_per_episode)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Summed Smoothed Reward per Episode')
plt.ylim([-1,11])
plt.xlim([-10, np.size(rewarded_image,0)*1.1])

plt.figure(4)
for idx in range (10):
    plt.subplot(2,5,idx+1)
    plt.plot(reward_pertrial_matrix[:, idx])
    if idx == 0:
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward per Trial %s' %(idx+1))
    else:
        plt.title('Trial %s' %(idx+1))
    plt.xlim([-10, np.size(rewarded_image,0)*1.1])
    plt.ylim([-.5, 1.5])
    plt.grid(True)
    plt.show()

b = 1

# smooth rewards
smoothing_coeff = 0.99
smooth_reward_pertrial = np.zeros([np.size(rewarded_image,0), 10])
smooth_reward_pertrial[0,:] = reward_pertrial_matrix[0,:]
for idx_episode in range(1,np.size(rewarded_image,0)):
    smooth_reward_pertrial[idx_episode, :] = smoothing_coeff*smooth_reward_pertrial[idx_episode-1, :] \
                                             + (1-smoothing_coeff)*reward_pertrial_matrix[idx_episode, :]

plt.figure(5)
for idx in range (10):
    plt.subplot(2,5,idx+1)
    plt.plot(smooth_reward_pertrial[:, idx])
    if idx == 0:
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Smooth Reward per Trial %s' %(idx+1))
    else:
        plt.title('Trial %s' %(idx+1))
    plt.xlim([-10, np.size(rewarded_image,0)*1.1])
    plt.ylim([-0.4, 1.5])
    plt.grid(True)
    plt.show()
#plt.savefig(fname)