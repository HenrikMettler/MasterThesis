import matplotlib
import pickle
import json
import time
import datetime

from keras.datasets import mnist
#from keras.callbacks import TensorBoard
#from helper import *
from functions import *
from Classes import *



filename_decisionnetwork = "decision_network2019-09-05 07:31:45.716316.pickle" # reference to the autoencoder file

infile = open(filename_decisionnetwork, 'rb')
policy_model, value_model, all_actions, all_rewards, all_states, all_trialIdx\
    = pickle.load(infile)
infile.close()

a = 1
