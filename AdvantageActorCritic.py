import random
import numpy as np
import keras.backend as K

#from lab import *
from tqdm import tqdm
from keras.models import Model
from keras import regularizers
from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten
from keras.optimizers import RMSprop
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Flatten, LSTM
from keras.optimizers import Adam


class AdvantageActorCritic:
    """ Advantage Actor Critic class
    """

    def __init__(self, act_dim, env_dim, input_size, num_units, gamma = 0.99, lr = 0.0001):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        #self.env_dim = env_dim
        self.gamma = gamma
        self.lr = lr
        # Create actor and critic networks
        self.shared = self.buildNetwork(input_size, num_units)
        self.actor = Actor(act_dim, self.shared, lr)
        self.critic = Critic(act_dim, self.shared, lr)
        # Build optimizers
        self.a_opt = self.actor.optimizer()
        self.c_opt = self.critic.optimizer()

    def buildNetwork(self, input_shape, num_units):
        """ Assemble top layers
        """
        lstm_layer = LSTM(num_units, return_sequences=True, input_shape=(1, input_shape))
        network = Sequential()
        network.add(lstm_layer)

        return network


    def policy_action(self, s):
        """ Use the actor to predict the next action to take, using the policy
        """
        return np.random.choice(np.arange(self.act_dim), 1, p=self.actor.predict(s).ravel())[0]

    def discount(self, r):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def train_models(self, states, actions, rewards, done):
        """ Update actor and critic networks from experience
        """
        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = self.discount(rewards)
        state_values = self.critic.predict(np.array(states))
        advantages = discounted_rewards - np.reshape(state_values, len(state_values))
        # Networks optimization
        self.a_opt([states, actions, advantages])
        self.c_opt([states, discounted_rewards])

    def train(self, env, args, num_episodes, summary_writer):
        """ Main A2C Training Algorithm
        """

        # Main Loop
        for e in range(num_episodes):

            # Reset episode
            time, cumul_reward, done = 0, 0, False
            old_state = env.reset()
            actions, states, rewards = [], [], []

            while not done:
                if args.render: env.render()
                # Actor picks an action (following the policy)
                a = self.policy_action(old_state)
                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done, _ = env.step(a)
                # Memorize (s, a, r) for training
                actions.append(to_categorical(a, self.act_dim))
                rewards.append(r)
                states.append(old_state)
                # Update current state
                old_state = new_state
                cumul_reward += r
                time += 1

            # Train using discounted rewards ie. compute updates
            self.train_models(states, actions, rewards, done)

        return

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)

class Agent:
    """ Agent Generic Class
    """

    def __init__(self, out_dim, lr):
        self.out_dim = out_dim
        self.rms_optimizer =  RMSprop(lr=lr, epsilon=0.1, rho=0.99)

    def fit(self, inp, targ):
        """ Perform one epoch of training
        """
        self.model.fit(self.reshape(inp), targ, epochs=1, verbose=0)

    def predict(self, inp):
        """ Critic Value Prediction
        """
        return self.model.predict(self.reshape(inp))

    def reshape(self, x):
        if len(x.shape) < 3: return np.expand_dims(x, axis=0)
        else: return x

class Actor(Agent):
    """ Actor for the A2C Algorithm
    """

    def __init__(self, out_dim, network, lr):
        Agent.__init__(self, out_dim, lr)
        self.model = self.addHead(network)
        self.action_pl = K.placeholder(shape=(None, self.out_dim))
        self.advantages_pl = K.placeholder(shape=(None,))

    def addHead(self, network):
        """ Assemble Actor network to predict probability of each action
        """
        x = Dense(128, activation='relu')(network.output) # not sure if this is appropriate
        out = Dense(self.out_dim, activation='softmax')(x)
        return Model(network.input, out)

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

    def save(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path)

class Critic(Agent):
    """ Critic for the A2C Algorithm
    """

    def __init__(self, out_dim, network, lr):
        Agent.__init__(self, out_dim, lr)
        self.model = self.addHead(network)
        self.discounted_r = K.placeholder(shape=(None,))

    def addHead(self, network):
        """ Assemble Critic network to predict value of each state
        """
        x = Dense(128, activation='relu')(network.output)
        out = Dense(1, activation='linear')(x)
        return Model(network.input, out)

    def optimizer(self):
        """ Critic Optimization: Mean Squared Error over discounted rewards
        """
        critic_loss = K.mean(K.square(self.discounted_r - self.model.output))
        updates = self.rms_optimizer.get_updates(self.model.trainable_weights, [], critic_loss)
        return K.function([self.model.input, self.discounted_r], [], updates=updates)

    def save(self, path):
        self.model.save_weights(path + '_critic.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
