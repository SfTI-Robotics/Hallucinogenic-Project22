import datetime
import os
import sys

import numpy as np

from agents.image_input import AbstractBrain
from agents.networks.actor_critic_networks import build_actor_cartpole_network, build_critic_cartpole_network, \
    build_critic_network, build_actor_network


# TODO: This implementation does not learn for Atari Games!
#   The initial policy heavily favours one action (99%)
#   After the first network-update, the favourite action has a probability of ~1.0, all others have ~0
#   Learner proceeds to only choose this one action, regardless of reward
#   implementation according to https://github.com/flyyufelix/VizDoom-Keras-RL

# TODO: add save and load function for actor and critic

class Learning(AbstractBrain.AbstractLearning):

    value_size = 1

    def __init__(self, observations, actions, config):
        super().__init__(observations, actions, config)

        if self.config['environment'] == 'CartPole-v1':
            self.actor_network = build_actor_cartpole_network(self.state_space, self.action_space, self.config['learning_rate'])
            self.critic_network = build_critic_cartpole_network(self.state_space, self.value_size, self.config['learning_rate'])
        else:
            self.actor_network = build_actor_network(self.state_space, self.action_space, self.config['learning_rate'])
            self.critic_network = build_critic_network(self.state_space, self.value_size, self.config['learning_rate'])

    def choose_action(self, state):
        policy = self.actor_network.predict(np.array([state])).flatten()
        return np.random.choice(np.arange(self.action_space), 1, p=policy)[0], policy[0]

    def discount_and_standardize_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.config['gamma'] + rewards[t]
            discounted_rewards[t] = running_add
        # Standardized discounted rewards
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards

    def train_network(self, states, actions, rewards, next_states, dones, step):
        # A2C is trained at the end of an episode, not after n steps
        episode_length = len(states)

        discounted_rewards = self.discount_and_standardize_rewards(rewards)

        # Prediction of state values for each state appears in the episode
        values = self.critic_network.predict(states)

        # Similar to one-hot target but the "1" is replaced by Advantage Function i.e. discounted_rewards R_t - Value
        advantages = np.zeros((episode_length, self.action_space))

        for i in range(episode_length):
            advantages[i][actions[i]] = discounted_rewards[i] - values[i]

        actor_loss = self.actor_network.fit(states, advantages, epochs=1, verbose=0)
        critic_loss = self.critic_network.fit(states, discounted_rewards, epochs=1, verbose=0)
        return actor_loss.history['loss'], critic_loss.history['loss']

    def save_network(self, save_path, model_name, timestamp=None):
        # create folder for model, if necessary
        if not os.path.exists(save_path + 'actors/'):
            os.makedirs(save_path + 'actors/')
        if not os.path.exists(save_path + 'critics/'):
            os.makedirs(save_path + 'critics/')
        if timestamp is None:
            timestamp = str(datetime.datetime.now())
        # save model weights
        self.actor_network.save_weights(save_path + 'actors/' + model_name + '_' + timestamp + '.h5', overwrite=True)
        self.critic_network.save_weights(save_path + 'critics/' + model_name + '_' + timestamp + '.h5', overwrite=True)

    def load_network(self, save_path, model_name) -> None:
        if os.path.exists(save_path + 'actors/') and os.path.exists(save_path + 'critics/'):
            self.actor_network.load_weights(save_path + 'actors/' + model_name)
            self.critic_network.load_weights(save_path + 'critics/' + model_name)
            print('Loaded model ' + model_name + ' from disk')
        else:
            sys.exit("Model can't be loaded. Model file " + model_name + " doesn't exist at " + save_path + ".")

    def get_test_learner(self):
        test_learner = Learning(self.state_space, self.action_space, self.config)
        # use current network weights for testing
        test_learner.actor_network.set_weights(self.actor_network.get_weights())
        test_learner.critic_network.set_weights(self.critic_network.get_weights())
        return test_learner
