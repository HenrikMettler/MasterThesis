from keras.layers import Input, Dense, Flatten, LSTM
from .critic import *
from .actor import *
from utils.networks import tfSummary
from utils.stats import gather_stats


class A2C:
    """ Class for running Advantage Actor Critic
    """

    def __init__(self, act_dim, env_dim, num_lstm_units, num_timesteps_unroll, gamma, lr, print_summary=False):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.num_lstm = num_lstm_units
        self.num_timesteps_unroll = num_timesteps_unroll
        self.gamma = gamma
        self.lr = lr
        self.print_summary = print_summary

        # Create actor and critic networks
        self.shared = self.buildNetwork()
        self.actor = Actor(self.env_dim, act_dim, self.shared, lr, print_summary)
        self.critic = Critic(self.env_dim, act_dim, self.shared, lr, print_summary)

        # Build optimizers
        self.a_opt = self.actor.optimizer()
        self.c_opt = self.critic.optimizer()

    def buildNetwork(self):
        """Build the parts of the model that are shared by Actor and Critic"""

        input_layer = Input(shape=(self.num_timesteps_unroll, self.env_dim))
        lstm_layer = LSTM(self.num_lstm)(input_layer)
        model = Model(input_layer, lstm_layer)

        if self.print_summary:
            model.summary()
        return model

    def select_action(self, state):
        """ Select next action by random selection of actor output
        """

        state = np.expand_dims(state,0) # expand 2D state (unroll_steps x dim) into 3D (1 sample x unroll_steps * dim)
        action_probs = self.actor.predict(state) # predict the probablility of each action
        selected_action = np.random.choice(action_probs.size, 1, p=action_probs.ravel()) # select one of the actions acc to their probability
        # log_prob = np.log(action_probs.squeeze(0)[selected_action])

        return selected_action, action_probs


    def discount_rewards(self, episode_rewards):
        """Discount rewards over an episode, using gamma"""
        discounted_rewards = np.zeros([np.size(episode_rewards, 0), 1])
        cumulated_rewards = 0
        for time in reversed(range(0, len(episode_rewards))):
            cumulated_rewards = episode_rewards[time] + cumulated_rewards * self.gamma
            discounted_rewards[time] = cumulated_rewards
        return discounted_rewards

    def train_models(self, states, actions, rewards, num_steps_unrolled = 10):
        """ Update actor and critic networks from experience
        """

        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = self.discount_rewards(rewards)
        state_values = self.critic.predict(states)
        advantages = discounted_rewards - state_values

        actions = np.squeeze(actions)
        advantages = np.squeeze(advantages)

        # Networks optimization
        self.a_opt([states, actions, advantages])
        self.c_opt([states, discounted_rewards])

        # Reset the internal states of the LSTM-layers in the network
        self.actor.model.reset_states()
        self.critic.model.reset_states()

        #squeezed_discounted_rewards = np.squeeze(discounted_rewards)
        #return squeezed_discounted_rewards

    # def save_weights(self, path):
    #     path += '_LR_{}'.format(self.lr)
    #     self.actor.save(path)
    #     self.critic.save(path)
    #
    # def load_weights(self, path_actor, path_critic):
    #     self.critic.load_weights(path_critic)
    #     self.actor.load_weights(path_actor)

