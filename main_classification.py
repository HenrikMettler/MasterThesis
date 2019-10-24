import pickle
import datetime
import time

from keras.models import Model
from keras import regularizers
from keras.datasets import mnist
from matplotlib import pyplot as plt
from multiprocessing import Pool

from functions import *
from AdvantageActorCritic import *

# Todo: 1) shape of input vector
#       2) reset of state after every episode


filename = "data/autoencoder_mnist2019-07-11 12:34:09.098789.pickle"

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
input_shape = 2*encoderModel.layers[-1].output_shape[1]*encoderModel.layers[-1].output_shape[2]*encoderModel.layers[-1].output_shape[3]

# Hyperparameters for training/testing
num_seeds = 1
num_episode_train = 1200 # Wang: 120'000
#num_episode_test = 300
num_trial_per_episode = 10
learning_rate = 7e-4
optimizer = 'rmsprop'
loss = 'mean_squared_error'

# LSTMCell parameters
num_units = 16 # Wang: 256
num_units_array = [16, 48, 256]
use_bias = True
# further LSTMCell parameters (so far set to default values -> move them up to parameter section when changing!
# kernel_initializer = 'glorot_uniform'
# recurrent_initializer = 'orthogonal'
# bias_initializer = 'zeros'
# unit_forget_bias = True
# kernel_regularizer = None
# recurrent_regularizer = None
# bias_regularizer = None
# kernel_constraint = None
# recurrent_constraint = None
# bias_constraint = None
# dropout = 0.0
# recurrent_dropout = 0.0
# implementation = 1
# activity_regularizer = None
# return_sequences = False
# return_state = False

# create networks
#action_network, value_network = create_networks(num_units, input_shape, optimizer, loss)
dm_network = create_dm_network(num_units, input_shape, optimizer, loss)


num_samples_per_training = 1
choice = np.zeros(num_samples_per_training)
prediction = np.zeros(num_samples_per_training)
target = np.zeros(num_samples_per_training)
all_inputs_for_training = np.zeros((num_samples_per_training, 1,input_shape))

num_train_loss = num_trial_per_episode/num_samples_per_training
train_loss_matrix = np.zeros((len(num_units_array), num_episode_train, num_train_loss))

time_seed = time.time()
time_episode = time.time()

idx_seed = 0
for num_units in num_units_array:
    print("This is the beginning of seed: ")
    print(idx_seed + 1)

    for idx_episode in range(num_episode_train):
        # print("This is the beginning of episode: ")
        # print(idx_episode+1)

        # sample two hidden representations
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
        reward, a, t = 0, 0, 0

        # perform the trials
        for idx_trial in range(num_trial_per_episode):

            idx_training_sample = idx_trial%num_samples_per_training


            # mix up the representations into random order #
            hidden_rep1 = hidden_representations[0]
            hidden_rep2 = hidden_representations[1]
            hidden_representations, labels = mix_up(hidden_rep1, hidden_rep2, labels)

            # define the target value of the trial
            if labels[0] == good_label:
                target [idx_training_sample] = 0
            elif labels[1] == good_label:
                target [idx_training_sample] = 1
            else:
                raise('Current architecture only supports 2 choices')


            choice[idx_training_sample], prediction[idx_training_sample], dm_input = trial_run(dm_network, hidden_representations, labels)
            all_inputs_for_training[idx_training_sample, :] = dm_input
            current_choice = choice[idx_training_sample]
            reward = check_reward(current_choice, labels, good_label)
            episode_reward.append(reward)

            if idx_training_sample == (num_samples_per_training-1): # only do training at the end of a block
                dm_network, loss_dict = training(dm_network, target, prediction, choice, all_inputs_for_training, optimizer, learning_rate)
                loss = loss_dict.get('loss', '')
                # loss matrix
                train_loss_matrix[idx_seed, idx_episode, idx_trial] = loss[0]

        # time_episode_over = time.time()
        # print("time for this episode: ")
        # print(time_episode_over - time_episode)
        # time_episode=time.time()



    time_seed_over = time.time()
    print("time for this seed: ")
    print(time_seed_over-time_seed)
    time_seed = time.time()
    idx_seed+=1


# Variable Saving
doSave = 1
if doSave == 1:
    currentTime = datetime.datetime.now()
    helperStr = ("Decision making %s" % currentTime)
    filename = (helperStr + ".pickle")
    with open(filename, "w") as f:  # The w stands for write
        pickle.dump([autoencoderModel, encoderModel, dm_network, train_loss_matrix, num_units], f)
    f.close()

# Plotting  Todo: make independent of upstream variable values

train_loss_seed_average = np.mean(train_loss_matrix, 0)
episode_average_counter = 100

# train_loss_seedAndEpisode_average = np.zeros([num_episode_train/episode_average_counter, num_trial_per_episode])
# for i in range(num_episode_train/episode_average_counter):
#     train_loss_seedAndEpisode_average[i,:] = np.mean(train_loss_seed_average[i*episode_average_counter:(i+1)*episode_average_counter-1,:],0)
#
#
# fig = plt.figure()
# num_plots = 8
# for j in range(num_plots):
#     plt.subplot(2,4,j+1)
#     episode_to_plot = j * 100 # this 10 is highly dep on upstream values!
#     tag = str(1 + j * 10000)
#     tag2 = str(1 + j * 10000 + episode_average_counter)
#     plt.plot(train_loss_seedAndEpisode_average[episode_to_plot, :])
#     plt.xlabel('Trial')
#     plt.ylabel('Error')
#     plt.title(['Av of Epi: ', tag, ' - ', tag2])
#     plt.grid(True)
#
# 