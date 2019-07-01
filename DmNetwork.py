# Source: https://keras.io/layers/recurrent/#rnn


import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import LSTM, LSTMCell, Dense
from keras.models import Sequential

import scipy.misc




# LSTMCell parameters
numUnits = 256
activation='tanh'
recurrent_activation='hard_sigmoid'
use_bias=True
# further LSTMCell parameters
kernel_initializer='glorot_uniform'
recurrent_initializer='orthogonal'
bias_initializer='zeros'
unit_forget_bias=True
kernel_regularizer=None
recurrent_regularizer=None
bias_regularizer=None
kernel_constraint=None
recurrent_constraint=None
bias_constraint=None
dropout=0.0
recurrent_dropout=0.0
implementation=1
# optional parameters in LSTM: set to default values according documentation (https://keras.io/layers/recurrent/#rnn)
# go_backwards=False
# stateful=False
# unroll=False





# Todo: Why are the params in cell and model??
lstmCells = LSTMCell(numUnits, activation=activation, recurrent_activation=recurrent_activation, use_bias=use_bias,
                         kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer,
                         bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias,
                         kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer,
                         bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint,
                         recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint,
                         dropout=dropout, recurrent_dropout=recurrent_dropout, implementation=implementation)

lstmNetwork = LSTM(lstmCells, activation=activation, recurrent_activation=recurrent_activation,
                  use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer,
                  bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias, kernel_regularizer=kernel_regularizer, recurrent_regularizer=None,
                  bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
                  bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False,
                  return_state=False)

model = Sequential()
model.add(lstmNetwork)
model.compile()
model.summary()
