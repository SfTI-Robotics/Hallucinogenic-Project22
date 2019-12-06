import copy
import random
from abc import ABC, abstractmethod
from collections import deque
import numpy as np


# TODO: Make different memories and differentiation prettier
#  (how to include policy? different add- and sample functions?)

class AbstractMemory(ABC):
    """
    stores tuples of (state, action, reward, next_state, done) for network-training
    """

    def __init__(self, capacity, state_space):
        self.state_space = state_space
        self.capacity = capacity
        self.tuples = deque(maxlen=capacity)

    def add_tuple(self, state, action, reward, next_state, done, policy=None) -> None:
        """
        Stores episode in memory
        :param state: state of episode
        :param action: action chosen by agent
        :param reward: reward received for state-action pair
        :param next_state: resulting state of environment
        :param done: flag, true if episode ended after action
        :param policy: policy that was used to choose the action
        """
        self.tuples.append((state, action, reward, next_state, done, policy))

    @abstractmethod
    def sample(self, processor, batch_size=None):
        """
        Samples data from memory
        :param batch_size: amount of random samples
        :param processor: processor used to convert samples from memory-format to network-format
        :return: states, actions, rewards, next_states, dones of randomly sampled episodes
        """
        pass

    def get_memory_size(self):
        return len(self.tuples)


class RandomBatchMemory(AbstractMemory):
    """
    stores tuples of (state, action, reward, next_state, done)
    following a first-in-first-out principle
    provides random fixed-size batches of tuples for network-training
    """

    def __init__(self, capacity, state_space):
        super().__init__(capacity, state_space)

    def sample(self, processor, batch_size=None):
        """
        Samples batch_size random episodes from memory
        :param processor: processor used to convert samples from memory-format to network-format
        :param batch_size: amount of random samples, default: amount of stored transition tuples
        :return: states, actions, rewards, next_states, dones of randomly sampled episodes
        """
        # update batch size in case memory doesn't contain enough values
        if batch_size is None:
            batch_size = len(self.tuples)
        else:
            batch_size = min(len(self.tuples), batch_size)

        batch = random.sample(self.tuples, batch_size)
        states = np.zeros((batch_size,) + self.state_space)
        next_states = np.zeros((batch_size,) + self.state_space)
        actions, rewards, dones = [], [], []

        for i in range(batch_size):
            states[i] = processor.process_state_for_network(batch[i][0])
            actions.append(batch[i][1])
            rewards.append(batch[i][2])
            next_states[i] = processor.process_state_for_network(batch[i][3])
            dones.append(batch[i][4])

        return states, actions, rewards, next_states, dones


class EpisodicMemory(AbstractMemory):
    """
    stores tuples of (state, action, reward, next_state, done)
    following a first-in-first-out principle
    tuples should be added in order of appearance in episode
    provides sets of all added tuples in order of addition for network-training
    tuples are removed from memory after sampling
    """

    def __init__(self, capacity, state_space, action_space):
        super().__init__(capacity, state_space)
        self.action_space = action_space

    def sample(self, processor, batch_size=None):
        batch = copy.deepcopy(self.tuples)
        batch_size = len(batch)
        states = np.zeros((batch_size,) + self.state_space)
        next_states = np.zeros((batch_size,) + self.state_space)
        actions, rewards, dones = [], [], []
        policies = np.zeros((batch_size, self.action_space))

        for i in range(batch_size):
            states[i] = processor.process_state_for_network(batch[i][0])
            actions.append(batch[i][1])
            rewards.append(batch[i][2])
            next_states[i] = processor.process_state_for_network(batch[i][3])
            dones.append(batch[i][4])
            policies[i] = batch[i][5]
        # remove all tuples from memory
        self.reset_memory()
        return states, actions, rewards, next_states, dones, policies

    def reset_memory(self) -> None:
        """
        Removes all tuples from memory
        """
        self.tuples = deque(maxlen=self.capacity)
