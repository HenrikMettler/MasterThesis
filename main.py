import matplotlib
import pickle
import datetime
#import tensorflow_probability as tfp

from keras.models import Model
from keras import regularizers
from keras.datasets import mnist
from keras import backend as K
from keras.callbacks import TensorBoard
from helper import *
from functions import *

matplotlib.use("TkAgg") # matplotlib import adapted for use on MacOS

from matplotlib import pyplot as plt

filename = "autoencoder_mnist2019-07-05 12:00:03.686985.pickle"

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

# Network parameters
num_threads = 32 # taken from Wang 2018

# Hyperparameters for training/testing
gamma = .9
a_size = 2
num_seeds = 1 # change to 8 or 10 or so
num_episode_train = 20000
num_episode_test = 300
num_trial_per_episode = 100
learning_rate = 7e-4
optimizer = 'rmsprop'
loss = 'categorical_crossentropy'

# LSTMCell parameters
num_units = 48 # Wang: 256
activation = 'tanh'
recurrent_activation = 'hard_sigmoid'
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

# create network
dm_network = create_dm_network(num_units, input_shape,optimizer, loss)
# worker = create_worker()

# coord = tf.train.Coordinator()
# sess = tf.Session()
episode_is_finished = 'false'

for idx_episode in range(num_episode_train):

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
    episode_buffer, episode_values, episode_frames, episode_reward = [],  [], [], []
    d = False
    reward, a, t = 0, 0, 0, 0

    # perform the trials
    for idx_trial in range(num_trial_per_episode):
        # mix up the representations into random order # Todo: This is a bit ugly, but doing it locally in the function doesn't work
        hidden_rep1 = hidden_representations[0,:,:,:]
        hidden_rep2 = hidden_representations[1,:,:,:]
        hidden_representations, labels = mix_up(hidden_rep1, hidden_rep2, labels)

        action = trial_run(dm_network, hidden_representations, labels)
        reward = check_reward(action, labels, good_label)
        episode_reward.append(reward)
        dm_network = training(dm_network, optimizer, learning_rate)


