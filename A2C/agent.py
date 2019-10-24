import numpy as np
from keras.optimizers import RMSprop


class Agent:
    """ Parent Class for actor and critic
    """

    def __init__(self, inp_dim, out_dim, lr, print_summary):
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.rms_optimizer = RMSprop(lr=lr, epsilon=0.1, rho=0.99)
        self.print_summary = print_summary

    def fit(self, inp, targ):
        """ Perform one epoch of training
        """
        self.model.fit(inp, targ, epochs=1, verbose=0)

