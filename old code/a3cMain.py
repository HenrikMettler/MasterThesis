""" Main script for the implementation of Asynchronous Advantage Actor Critic (A3C)
"""

import os
import sys
import gym
import numpy as np
import pandas as pd
import tensorflow as tf
import functions

from A3C.a3c import A3C
from functions import Create_Args


from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical
from keras.optimizers import RMSprop

from utils.continuous_environments import Environment
from utils.networks import get_session

gym.logger.set_level(40)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(args=None):

    # Todo: this needs to be changed, just here for the logic
    num_hiddenUnits = 100

    # main parameters
    nb_episodes = 20000  # number of train episodes
    batch_size = 64  # Todo: check in Wang 2017
    consecutive_frames = 4  # Todo: check in Wang 2017
    training_interval = 30  # Todo: check in Wang 2017
    n_threads = 32
    gamma = 0.91
    lr = 0.00075
    entropy_cost = 0.001
    sv_estimate_cost = 0.4
    optimizer = RMSprop(lr=lr)  # if the optimizers is to be changed, the new one need to be imported
    n_timeSteps = 100
    dataset_size = 2 # Todo: check if this is correct, I reckon it must be the number of diff inputs

    # option statistics
    gather_stats = 'false'
    render = 'false'
    env = 'empty'  #ToDo: implement harlow()

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
                       kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout,
                       implementation,
                       activity_regularizer, return_sequences, return_state]

    args = Create_Args(nb_episodes, batch_size, consecutive_frames, training_interval, n_threads, gamma, lr, optimizer, n_timeSteps, gather_stats, render, env)

    set_session(get_session())
    summary_writer = tf.summary.FileWriter('A3C'+ "/tensorboard_" + args.env)

    # Environment Initialization # Todo create Harlow env - indep of gym??
    #env = harlow()(gym.make(args.env), args.consecutive_frames)
    #env.reset()
    env_dim =  (num_hiddenUnits,) #env.get_state_size() # Todo: replace, understand dim
    action_dim = np.int32(2)  #gym.make(args.env).action_space.n

    # create A3C instance
    algo = A3C(action_dim, env_dim, args.consecutive_frames, lstm_param_list, dataset_size, args.gamma, args.lr, args.optimizer, args.n_timeSteps)

    # Train
    stats = algo.train(args, summary_writer)

    # Export results to CSV
    if(args.gather_stats):
        df = pd.DataFrame(np.array(stats))
        df.to_csv(args.type + "/logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

    # Save weights and close environments
    exp_dir = '{}/models/'.format(args.type)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    export_path = '{}{}_ENV_{}_NB_EP_{}_BS_{}'.format(exp_dir,
        args.type,
        args.env,
        args.nb_episodes,
        args.batch_size)

    algo.save_weights(export_path)
    args.env.close()

if __name__ == "__main__":
    main()