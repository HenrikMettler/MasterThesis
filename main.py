import numpy as np
import matplotlib
import pickle
import datetime

from keras.layers import Conv2D, Dense, Input,  MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import regularizers
from keras.datasets import mnist
from keras import backend as K
from keras.callbacks import TensorBoard
from helper import *
from functions import *
from keras.optimizers import RMSprop

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
(x_train, _), (x_test, _) = mnist.load_data()

# Prepare Input: Normalizing, Flatten Images
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# Network parameters
num_threads = 32 # taken from Wang 2018

# Hyperparameters for training/testing
gamma = .9
a_size = 2
num_seeds = 8
num_episode_train = 20000
num_episode_test = 300
learning_rate = 7e-4
optimizer = RMSprop(lr=learning_rate) # if the optimizers is to be changed, the new one need to be imported

# LSTMCell parameters
num_units = 256
activation = 'tanh'
recurrent_activation = 'hard_sigmoid'
use_bias = True
# further LSTMCell parameters (so far set to default values -> move them up to parameter section when changing!
kernel_initializer = 'glorot_uniform'
recurrent_initializer = 'orthogonal'
bias_initializer = 'zeros'
unit_forget_bias = True
kernel_regularizer = None
recurrent_regularizer = None
bias_regularizer = None
kernel_constraint = None
recurrent_constraint = None
bias_constraint = None
dropout = 0.0
recurrent_dropout = 0.0
implementation = 1
activity_regularizer = None
return_sequences = False
return_state = False

# set arguments to a list
# DO NOT change the order of these arguments, as they are used index wise in create_dm_network
lstm_param_list = [num_units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer,
                   bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer,
                   kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, implementation,
                   activity_regularizer, return_sequences, return_state]

#Todo: Check the below initializations
name = "worker_0"
update_local_ops = update_target_graph('global', name)
environment_name = 'harlow_mnist'  # options: harlow_mnist, (harlow_object)

train_mode = 1

for idx_seed in range(num_seeds):
    # create network, worker and task environment
    dm_network = create_dm_network(lstm_param_list, encoderModel, optimizer) # Todo: only input the layer, not the whole model
    # worker = create_worker()
    environment = create_environment(environment_name)

    episode_count = 0

    # initialize session
    # set the seed
    np.random.seed(idx_seed)
    tf.set_random_seed(idx_seed)

    coord = tf.train.Coordinator()
    sess = tf.Session()
    episode_is_finished = 0
    while episode_count < num_episode_train & episode_is_finished == 0:
        sess.run(update_local_ops)
        episode_buffer = []
        episode_values = []
        episode_frames = []
        episode_reward = 0
        episode_step_count = 0
        d = False
        r = 0
        a = 0
        t = 0
        s = environment.reset()
        rnn_state = dm_network.state_init





        # if in train mode: do training
        if train_mode == 1:
            trainer = training(trainer,dm_network,optimizer,learning_rate)
        else: # is in test -> plot some stuff? (Todo)
            a = 1





        # evaluate if episode is terminated

        # update episode count
        if episode_is_finished == 0:
            episode_count += 1
        else:
            episode_count = 0
