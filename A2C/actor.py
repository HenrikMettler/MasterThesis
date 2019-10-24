import numpy as np
import keras.backend as K

from keras.models import Model, load_model
from keras.layers import Dense
from .agent import Agent


class Actor(Agent):
    """ Actor for the A2C Algorithm
    """

    def __init__(self, inp_dim, out_dim, network, lr, print_summary):
        Agent.__init__(self, inp_dim, out_dim, lr, print_summary)
        self.model = self.addHead(network)
        self.action_pl = K.placeholder(shape=(None, self.out_dim))
        self.advantages_pl = K.placeholder(shape=(None,))
        #self.states = K.placeholder(shape=(None, inp_dim))

    def addHead(self, network):
        """ Assemble Actor network to predict probability of each action
        """
        lstm_out_size = network.output_shape[1]
        x = Dense(lstm_out_size, activation='relu')(network.output)
        out = Dense(self.out_dim, activation='softmax')(x)
        model = Model(network.input, out)
        if self.print_summary:
            model.summary()
        return model

    def optimizer(self):
        """ Actor Optimization: Advantages + Entropy term to encourage exploration
        (Cf. https://arxiv.org/abs/1602.01783)
        """
        weighted_actions = K.sum(self.action_pl * self.model.output, axis=1)
        eligibility = K.log(weighted_actions + 1e-10) * K.stop_gradient(self.advantages_pl)
        entropy = K.sum(self.model.output * K.log(self.model.output + 1e-10), axis=1)
        loss = 0.001 * entropy - K.sum(eligibility)

        updates = self.rms_optimizer.get_updates(self.model.trainable_weights, [], loss)
        return K.function([self.model.input, self.action_pl, self.advantages_pl], [], updates=updates)

    def predict(self, state):
        """ Predict the action value outputs"""
        predicted_state_value = self.model.predict(state)
        return predicted_state_value

    def save(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
