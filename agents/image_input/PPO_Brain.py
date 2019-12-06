import datetime
import os
import sys
from typing import Tuple

import numpy as np

import agents.image_input.AbstractBrain as AbstractBrain
from agents.networks.ppo_networks import build_ppo_critic_network, build_ppo_actor_network, \
    build_ppo_critic_cartpole_network, build_ppo_actor_cartpole_network


# see https://github.com/OctThe16th/PPO-Keras/blob/master/Main.py


class Learning(AbstractBrain.AbstractLearning):

    def __init__(self, observations, actions, config):
        super().__init__(observations, actions, config)
        # initialize networks for non-image data, e.g. gym classic control environments
        if self.config['environment'] == 'CartPole-v1':
            self.actor_network \
                = build_ppo_actor_cartpole_network(self.state_space, self.action_space, self.config['learning_rate'],
                                                   self.config['clipping_loss_ratio'])  # , self.config['entropy_loss_ratio'])
            self.critic_network = build_ppo_critic_cartpole_network(self.state_space, self.config['learning_rate'])
        # initialize networks for image data, e.g gym atari environments
        else:
            self.actor_network \
                = build_ppo_actor_network(self.state_space, self.action_space, self.config['learning_rate'],
                                          self.config['clipping_loss_ratio'])  # , self.config['entropy_loss_ratio'])
            self.critic_network = build_ppo_critic_network(self.state_space, self.config['learning_rate'])
        # placeholder for action and value, used during actor-prediction
        self.dummy_action, self.dummy_value = np.zeros((1, self.action_space)), np.zeros((1, 1))
        # stores policies, used during network updates

    def choose_action(self, state) -> Tuple[int, np.ndarray]:
        # get policy of current network
        policy = self.actor_network.predict([np.array([state]), self.dummy_value, self.dummy_action])
        # choose action, use policy as weights
        return np.random.choice(self.action_space, p=np.nan_to_num(policy[0])), policy[0]

    def compute_advantage(self, states, rewards, next_states, dones) -> np.ndarray:
        """
        Computes advantages of actions to update actor
        How much better or worse was actual reward, compared to our expectations?
        See equations (11) and (12) in 'Proximal Policy Optimization Algorithms'
        :param states: list of states
        :param rewards: rewards received in states
        :param next_states: states reached after reward was received
        :param dones: flag, true if episode was completed after reward was received
        :return: advantages as ndarray
        """
        advantages = np.zeros_like(rewards)
        previous_advantage = 0
        next_values = self.critic_network.predict(next_states)
        current_values = self.critic_network.predict(states)
        # advantage at time step t: current advantage + discounted following advantages
        for t in reversed(range(len(rewards))):
            if dones[t]:
                previous_advantage = 0
                current_advantage = rewards[t] + - current_values[t]
            else:
                # How good was my current action (following my policy from here on) compared to my expectations?
                current_advantage = rewards[t] + self.config['gamma'] * next_values[t] - current_values[t]
            advantages[t] = current_advantage + self.config['gamma'] * self.config['lambda'] * previous_advantage
            previous_advantage = advantages[t]
        return advantages

    def compute_critic_targets(self, rewards, next_states, dones) -> np.ndarray:
        """
        Computes target values to update critic
        :param rewards: rewards received after previous actions
        :param next_states: states reached after receiving reward
        :param dones: flag, true if episode was completed after reward was received
        :return: target values as ndarray
        """
        targets = np.zeros_like(rewards)
        next_values = self.critic_network.predict(next_states)
        # updates computed like in dqn
        for i in range(len(rewards)):
            if dones[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = rewards[i] + self.config['gamma'] * next_values[i]
        return targets

    def train_network(self, states, actions, rewards, next_states, dones, policies) -> None:
        # compute advantages and target values
        advantages = self.compute_advantage(states, rewards, next_states, dones)
        # TODO: test this for Atari environments
        # normalize advantage (might work better in some cases
        # advantage = (advantage - advantage.mean()) / advantage.std()
        targets = self.compute_critic_targets(rewards, next_states, dones)
        # construct np arrays of actions taken
        actions_taken = np.zeros((len(dones), self.action_space))
        for i in range(len(dones)):
            actions_taken[i][actions[i]] = 1
        # train networks
        self.actor_network.fit([states, advantages, policies], [actions_taken],
                               batch_size=self.config['batch_size'], shuffle=True, epochs=self.config['epochs'],
                               verbose=False)
        self.critic_network.fit([states], [targets], batch_size=self.config['batch_size'], shuffle=True,
                                epochs=self.config['epochs'], verbose=False)
        self.old_predictions = []

    def save_network(self, save_path, model_name, timestamp=None) -> None:
        # create folder for model, if necessary
        if not os.path.exists(save_path + 'actors/'):
            os.makedirs(save_path + 'actors/')
        if not os.path.exists(save_path + 'critics/'):
            os.makedirs(save_path + 'critics/')
        # set timestamp if none was specified
        if timestamp is None:
            timestamp = str(datetime.datetime.now())
        # save model weights
        self.actor_network.save_weights(save_path + 'actors/' + model_name + '_' + timestamp + '.h5', overwrite=True)
        self.critic_network.save_weights(save_path + 'critics/' + model_name + '_' + timestamp + '.h5', overwrite=True)

    def load_network(self, save_path, model_name) -> None:
        if os.path.exists(save_path + 'actors/') and os.path.exists(save_path + 'critics/'):
            # load network weights from files
            self.actor_network.load_weights(save_path + 'actors/' + model_name)
            self.critic_network.load_weights(save_path + 'critics/' + model_name)
            print('Loaded model ' + model_name + ' from disk')
        else:
            sys.exit("Model can't be loaded. Model file " + model_name + " doesn't exist at " + save_path + ".")

    def get_test_learner(self) -> 'Learning':
        # initialize test-learner
        test_learner = Learning(self.state_space, self.action_space, self.config)
        # use current network weights for testing
        test_learner.actor_network.set_weights(self.actor_network.get_weights())
        test_learner.critic_network.set_weights(self.critic_network.get_weights())
        return test_learner
