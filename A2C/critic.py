import numpy as np
import keras.backend as K

from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam
from .agent import Agent


class Critic(Agent):
    """ Critic for the A2C Algorithm
    """

    def __init__(self, inp_dim, out_dim, network, lr, print_summary):
        Agent.__init__(self, inp_dim, out_dim, lr, print_summary)
        self.model = self.addHead(network)
        self.discounted_r = K.placeholder(shape=(None,))
        #self.states = K.placeholder(shape=(None, inp_dim))

    def addHead(self, network):
        """ Assemble Critic network to predict value of each state
        """
        lstm_out_size = network.output_shape[1]
        x = Dense(lstm_out_size, activation='relu')(network.output)
        out = Dense(1, activation='linear')(x)
        model = Model(network.input, out)
        if self.print_summary:
            model.summary()
        return model

    def optimizer(self):
        """ Critic Optimization: Mean Squared Error over discounted rewards
        """
        critic_loss = K.mean(K.square(self.discounted_r - self.model.output))
        updates = self.rms_optimizer.get_updates(self.model.trainable_weights, [], critic_loss)
        return K.function([self.model.input, self.discounted_r], [], updates=updates)

    def predict(self, inp):
        """ Critic Value Prediction
        """
        if inp.ndim == 2:
            inp = np.expand_dims(inp, 0)
        predicted_state_value = self.model.predict(inp)
        return predicted_state_value

    def save(self, path):
        self.model.save_weights(path + '_critic.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
