import pickle
import datetime
import time

from keras.models import Model
from keras import regularizers
from keras.datasets import mnist
from multiprocessing import Pool

from functions import *
from AdvantageActorCritic import *
from classification_sample import *
from classification_episode import *

num_units_array = [16, 48, 256, 1028]
learning_rates = [1e-4, 2e-4, 5e-4, 7e-4, 1e-3, 2e-3]
swaps = [1, 0]
#num_seeds = 10

# for num_units in num_units_array:
#     for learning_rate in learning_rates:
#         for swap in swaps:
#             classification_sample(num_units, learning_rate, swap)
#             classification_episode(num_units, learning_rate, swap)

for learning_rate in learning_rates:
    classification_sample(num_units = 48, learning_rate=learning_rate, swap=1)

classification_sample(num_units=48, learning_rate=learning_rate, swap=0)