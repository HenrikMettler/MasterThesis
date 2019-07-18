import matplotlib
import pickle

from keras.models import Model
from keras import regularizers
from keras.datasets import mnist
from keras import backend as K
from keras.callbacks import TensorBoard
from helper import *
from functions import *

matplotlib.use("TkAgg") # matplotlib import adapted for use on MacOS

from matplotlib import pyplot as plt


filename= "Decision making 2019-07-18 11:31:39.858836.pickle"

infile = open(filename,'rb')
autoencoderModel, encoderModel, dm_network, train_loss_matrix, num_units = pickle.load(infile)


train_loss_seed_average = np.mean(train_loss_matrix, 0)
episode_average_counter = 100

train_loss_seedAndEpisode_average = np.zeros([np.shape(train_loss_matrix,1)/episode_average_counter, np.shape(train_loss_matrix,2)])
for i in range(np.shape(train_loss_matrix,1)/episode_average_counter):
    train_loss_seedAndEpisode_average[i,:] = np.mean(train_loss_seed_average[i*episode_average_counter:(i+1)*episode_average_counter-1,:],0)


fig = plt.figure()
num_plots = 8
for j in range(num_plots):
    plt.subplot(2,4,j+1)
    episode_to_plot = j * 100 # this 10 is highly dep on upstream values!
    tag = str(1 + j * 10000)
    tag2 = str(1 + j * 10000 + episode_average_counter)
    plt.plot(train_loss_seedAndEpisode_average[episode_to_plot, :])
    plt.xlabel('Trial')
    plt.ylabel('Error')
    plt.title(['Av of Epi: ', tag, ' - ', tag2])
    plt.grid(True)


