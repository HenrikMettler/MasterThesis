from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import numpy as np

filename = "data/Class_sample_1011_nEpisode:120000_nUnits:48_learning_rate:0.001_swap:1.pickle"
infile = open(filename, "rb")
[dm_network, train_loss_matrix, prediction_matrix, choice_matrix, target_matrix] = pickle.load(infile)


train_loss_start = np.squeeze(train_loss_matrix[0,0:100,:])
mean_train_loss_start = np.mean(train_loss_start,0)

train_loss_20000 = np.squeeze(train_loss_matrix[0,20000:20100,:])
mean_train_loss_20000 = np.mean(train_loss_20000,0)

train_loss_end = np.squeeze(train_loss_matrix[0,119900:,:])
mean_train_loss_end = np.mean(train_loss_end,0)

choice_start = np.squeeze(choice_matrix[0,0:100,:])
target_start = np.squeeze(target_matrix[0,0:100,:])

classification = np.ones([120000,10])
diff_choice_target = np.squeeze(np.abs([target_matrix-choice_matrix]))
classification = classification - diff_choice_target

smoothing_coeff = 0.999
smoothed_classification = np.zeros([120000,10])
smoothed_classification[0,:] = np.mean(classification[0:100,:],0)
smoothed_loss = np.zeros([120000,10])
smoothed_loss[0,:] =np.mean(train_loss_matrix[0,0:100,:],0)


for idx_episode in range(1, 120000):
    smoothed_classification[idx_episode, :] = smoothing_coeff * smoothed_classification[idx_episode - 1, :] \
                                              + (1-smoothing_coeff) * classification[idx_episode, :]
    smoothed_loss[idx_episode,:] = smoothing_coeff * smoothed_loss[idx_episode-1,:]  +\
                                   (1-smoothing_coeff) * train_loss_matrix[0,idx_episode, :]


plt.figure(1)
for idx_trial in range(10):
    plt.plot(smoothed_classification[:,idx_trial])

plt.figure(2)
for idx_trial in range(10):
    plt.plot(smoothed_loss[:,idx_trial])