import numpy as np
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.models import Model
from keras import regularizers
from keras.utils import plot_model
import matplotlib
matplotlib.use("TkAgg") # matplotlib import adapted for use on MacOS
from matplotlib import pyplot as plt

# Import data
# TODO: Define input data set
sequence = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# Prepare Input: Reshape
n_in = len(sequence)
sequence = sequence.reshape((1, n_in, 1))


#  Network Parameters, Train Parameters


# Regularization Parameter


# Create Dependant Parameters
inputDim = n_in

# Create Model
autoencoderLstm = Sequential()
autoencoderLstm.add


# Train Model
autoencoderLstm.fit(x_train, x_train, epochs=num_epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, x_test))



# plot results