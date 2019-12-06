import datetime
import os
import sys

import numpy as np

from agents.image_input import AbstractBrain
from agents.networks.actor_critic_networks import \
    build_actor_network, build_actor_cartpole_network


# see https://github.com/ShuvenduBikash/Deep-reinforcement-learning/blob/master/policy_gradient/cart-pole%20keras.py
#     https://github.com/simoninithomas/reinforcement-learning-1


class Learning(AbstractBrain.AbstractLearning):

    def __init__(self, observations, actions, config):
        super().__init__(observations, actions, config)

        if self.config['environment'] == 'CartPole-v1':
            self.network = \
                build_actor_cartpole_network(self.state_space, self.action_space, self.config['learning_rate'])
        else:
            self.network = build_actor_network(self.state_space, self.action_space, self.config['learning_rate'])
        self.all_predictions = []

    def choose_action(self, state):
        policy = self.network.predict(np.array([state])).flatten()
        return np.random.choice(np.arange(self.action_space), 1, p=policy)[0], policy[0]

    def discount_and_standardize_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            running_add = running_add * self.config['gamma'] + rewards[t]
            discounted_rewards[t] = running_add

        # Standardized discounted rewards
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards

    def train_network(self, states, actions, rewards, next_states, dones, step):
        discounted_rewards = self.discount_and_standardize_rewards(np.vstack(rewards))

        advantages = np.zeros((len(dones), self.action_space))
        for i in range(len(dones)):
            advantages[i][actions[i]] = discounted_rewards[i]
        self.network.fit(states, advantages, epochs=1, verbose=0)

    def save_network(self, save_path, model_name, timestamp=None):
        # create folder for model, if necessary
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # set timestamp if none was specified
        if timestamp is None:
            timestamp = str(datetime.datetime.now())
        # save model weights
        self.network.save_weights(save_path + model_name + '_' + timestamp + '.h5', overwrite=True)

    def load_network(self, save_path, model_name) -> None:
        if os.path.exists(save_path):
            self.network.load_weights(save_path + model_name)
            print('Loaded model ' + model_name + ' from disk')
        else:
            sys.exit("Model can't be loaded. Model file " + model_name + " doesn't exist at " + save_path + ".")

    def get_test_learner(self):
        test_learner = Learning(self.state_space, self.action_space, self.config)
        # use current network weights for testing
        test_learner.network.set_weights(self.network.get_weights())
        return test_learner
