from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class AbstractLearning(ABC):

    # values used for plotting
    epsilon = 0
    e_greedy_formula = ''

    def __init__(self, observations, actions, config):
        self.state_space = observations
        self.action_space = actions
        self.config = config

    @abstractmethod
    def choose_action(self, state) -> Tuple[int, np.ndarray]:
        """
        Chooses action to take in the given state, depending on the current policy
        :param state: state-object as ndarray
        :return: int-value representing action, policy
        """
        pass

    @abstractmethod
    def train_network(self, states, actions, rewards, next_states, dones, step) -> None:
        """
        Trains network(s)
        :param states: array of state-objects
        :param actions: list of actions taken in states
        :param rewards: list of rewards received for actions
        :param next_states: array of state-objects, states reached after executing action
        :param dones: flag, true if episode ended after action
        :param step: current training step
        """
        pass

    @abstractmethod
    def save_network(self, save_path, model_name, timestamp=None) -> None:
        """
        Saves current model to .h5 file, overrides previous model for same environment and algorithm
        :param save_path: path to model folder
        :param model_name: name of model file
        :param timestamp: optional timestamp, if none is specified, current time is used
        """
        pass

    @abstractmethod
    def load_network(self, save_path, model_name) -> None:
        """
        Loads previously saved model file to learner.network
        :param save_path: path to model folder
        :param model_name: name of model file
        """
        pass

    @abstractmethod
    def get_test_learner(self) -> 'AbstractLearning':
        """
        Creates a version of the current learner that can be used for testing
        (with the same model weights, but without disrupting any data of the original agent)
        :return: duplicate of the current learner that can be used for testing
        """
        pass
