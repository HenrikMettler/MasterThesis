import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Flatten, LSTM


def generate_data(n=10):

    dataSet = []
    for index in range(n*n-1):
        first_index = np.floor((index+1)/n)
        first_index = first_index.astype(int)
        second_index = (index+1) % n
        if first_index != second_index:
            dataSet.append([first_index, second_index])

    return dataSet


def build_decision_network(num_units=4, optimizer='rmsprop', loss='mean_squared_error', output_activation='sigmoid'):

    lstm_layer = LSTM(num_units, return_sequences='true', input_shape=(1, 2))

    network = Sequential()
    network.add(lstm_layer)
    network.add(Dense(1, activation=output_activation))
    network.compile(optimizer=optimizer, loss=loss)
    network.summary()

    return network

def select_data_for_episode(dataSet):

    """selects a data sample and its mirror for an Episode + the correct label"""
    i = np.random.randint(low=0, high=np.size(dataSet,0), size=1)
    data_sample_1 = dataSet[i[0]]
    data_sample_2=[]
    data_sample_2.append(data_sample_1[1])
    data_sample_2.append(data_sample_1[0])
    i = np.random.random_sample()
    j = np.round(i)
    j = j.astype(int)
    correct_label = data_sample_1[j]

    return data_sample_1, data_sample_2, correct_label


def pick_one_sample(data_sample_1, data_sample_2):

    i = np.random.random_sample()
    if i < 0.5:
        sample = data_sample_1
    else:
        sample = data_sample_2

    return sample
