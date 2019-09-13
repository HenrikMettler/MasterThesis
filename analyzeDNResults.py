import matplotlib
import pickle
import numpy as np
import seaborn as sns

#matplotlib.use("Agg") # matplotlib import adapted for use on MacOS

from matplotlib import pyplot as plt


import matplotlib
import pickle
import numpy as np
matplotlib.use("TkAgg") # matplotlib import adapted for use on MacOS

from matplotlib import pyplot as plt

# load episode durations from short runs
episode_durations = []
for idxSet in range(1,10):
    helperStr = ("episode_durations%s" % idxSet)
    filename = (helperStr + ".pickle")
    infile = open(filename, "rb")
    current_durations = pickle.load(infile)
    episode_durations.append(current_durations)

x = np.arange(1, 3001)
mean_episode_duration = np.mean(episode_durations,0)
std_episode_duration = np.std(episode_durations,0)

# Load full run data from 5 runs
data_strucutres = []
for idxSet in range(5):
    helperStr = ("decisionNetwork0909_setNr%s" % idxSet)
    filename = (helperStr + ".pickle")
    infile = open(filename, "rb")
    current_datastructure = pickle.load(infile)
    data_strucutres.append(current_datastructure)


# compute summed rewards and reward in second trial
summed_reward = np.zeros([5, 1200])
trial_reward_matrix = np.zeros([10,5,1200])

first_trial_reward = np.zeros([5, 1200])
second_trial_reward = np.zeros([5, 1200])
third_trial_reward = np.zeros([5, 1200])
fourth_trial_reward = np.zeros([5, 1200])
fifth_trial_reward = np.zeros([5, 1200])
sixth_trial_reward = np.zeros([5, 1200])
seventh_trial_reward = np.zeros([5, 1200])
eight_trial_reward = np.zeros([5, 1200])
ninth_trial_reward = np.zeros([5, 1200])
tenth_trial_reward = np.zeros([5, 1200])
num_timestep = np.zeros([5, 1200])

for idxSet in range(5):
    for idxEpisode in range(1200):
        num_timestep[idxSet, idxEpisode] = len(data_strucutres[idxSet].rewards[idxEpisode])
        current_rewards = data_strucutres[idxSet].rewards[idxEpisode]
        current_trialindices = data_strucutres[idxSet].trialidx[idxEpisode]
        for idxTime in range(np.int(num_timestep[idxSet, idxEpisode])):
            summed_reward[idxSet, idxEpisode] += current_rewards[idxTime]
            current_trialidx = current_trialindices[idxTime]
            trial_reward_matrix[current_trialidx,idxSet, idxEpisode] += current_rewards[idxTime]

            # if current_trialindices[idxTime] == 0:
            #     first_trial_reward[idxSet, idxEpisode] += current_rewards[idxTime]
            # if current_trialindices[idxTime] == 1:
            #     second_trial_reward[idxSet, idxEpisode] += current_rewards[idxTime]
            # if current_trialindices[idxTime] == 2:
            #     third_trial_reward[idxSet, idxEpisode] += current_rewards[idxTime]
            # if current_trialindices[idxTime] == 3:
            #     fourth_trial_reward[idxSet, idxEpisode] += current_rewards[idxTime]
            # if current_trialindices[idxTime] == 4:
            #     fifth_trial_reward[idxSet, idxEpisode] += current_rewards[idxTime]
            # if current_trialindices[idxTime] == 5:
            #     sixth_trial_reward[idxSet, idxEpisode] += current_rewards[idxTime]
            # if current_trialindices[idxTime] == 6:
            #     seventh_trial_reward[idxSet, idxEpisode] += current_rewards[idxTime]
            # if current_trialindices[idxTime] == 7:
            #     eight_trial_reward[idxSet, idxEpisode] += current_rewards[idxTime]
            # if current_trialindices[idxTime] == 7:
            #     ninth_trial_reward[idxSet, idxEpisode] += current_rewards[idxTime]
            # if current_trialindices[idxTime] == 7:
            #     second_trial_reward[idxSet, idxEpisode] += current_rewards[idxTime]

        summed_reward[idxSet, idxEpisode] = summed_reward[idxSet,idxEpisode]/num_timestep[idxSet, idxEpisode]

a = 1




# plotting episode durations
# plt.figure(1)
# for idx in range(9):
#     plt.plot(episode_durations[idx])
# plt.xlabel('Episode')
# plt.ylabel('Duration')
# plt.title('Num timesteps per Episode for 9 Agents')
# plt.xlim([-100, 3100])
# plt.ylim([-5, 225])
# plt.grid(True)
# plt.show()
#
#
# plt.figure(2)
# plt.errorbar(x, y=mean_episode_duration, yerr=std_episode_duration)
# plt.xlabel('Episode')
# plt.ylabel('Duration')
# plt.title('Num timesteps per Episode for 9 Agents')
# plt.xlim([-100, 3100])
# plt.ylim([-5, 225])
# plt.grid(True)
# plt.show()

plt.figure(3)
for idx in range(9):
    plt.subplot(3,3,idx+1)
    plt.plot(episode_durations[idx])
    plt.xlabel('Episode')
    plt.ylabel('Time steps per episode')
    title_string = 'Agent: %s' %(idx+1)
    plt.title(title_string)
    plt.xlim([-100, 3100])
    plt.ylim([-5, 225])
    plt.grid(True)
    plt.show()

plt.figure()

plt.subplot(2,1,1)
for idx in range(5):
    plt.plot(summed_reward[idx])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Average Reward for 5 Agents')
plt.xlim([-100, 1300])
plt.ylim([-0.1, 0.5])
plt.grid(True)
plt.show()



mean_episode_duration = np.mean(episode_durations,0)

plt.subplot(2,1,2)
plt.errorbar(episode_durations)
plt.xlabel('Episode')
plt.ylabel('Duration')
plt.title('Average duration')
plt.xlim([-100, 3100])
plt.ylim([-5, 225])
plt.grid(True)
plt.show()

a = 1

extract_episode1 = episode_durations[0][1250:1400]


# Plot Trial Rewards
for idxSet in range (5):
    plt.figure(10+idxSet)
    for idxTrial in range(10):
        plt.subplot(2,5, idxTrial+1)
        plt.plot(trial_reward_matrix[idxTrial, idxSet, :])
        plt.xlim([-100, 1300])
        plt.ylim([-0.5, 1.5])
        plt.grid(True)
        plt.show()