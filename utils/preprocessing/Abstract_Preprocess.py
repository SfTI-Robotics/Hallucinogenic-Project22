from abc import ABC
from collections import deque

import numpy as np
from PIL import Image


# see https://github.com/rohitgirdhar/Deep-Q-Networks for preprocessing


class AbstractProcessor(ABC):

    # game-specific values
    step_max = None
    reward_min = None
    reward_max = None

    def __init__(self):
        self.deque = deque(maxlen=4)
        # initialize deque with empty default state to make sure shape of stored state always stays the same
        for i in range(4):
            self.deque.append(np.zeros((84, 84), dtype=np.int))
        self.resize_size = (110, 84)
        self.standardize_image = True
        self.new_size = 84

    def get_state_space(self) -> tuple:
        """
        Get shape of processed state
        :return: shape
        """
        return np.shape(self.deque)

    def process_state_for_memory(self, state, is_new_episode) -> np.ndarray:
        """
        Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eighth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        :param state: state as given by the gym-environment
        :param is_new_episode: flag, true if state is first state in episode
        :return: state in memory-format
        """
        # convert image to greyscale, downsize
        state_image = Image.fromarray(state, 'RGB')
        state_image = state_image.convert('L')  # to gray
        state_image = state_image.resize((self.new_size, self.new_size), Image.ANTIALIAS)
        state_image = np.array(state_image).astype('uint8')

        if is_new_episode:
            # fill deque with first state of episode
            self.deque.append(state_image)
            self.deque.append(state_image)
            self.deque.append(state_image)
            self.deque.append(state_image)
        else:
            # append new frame to deque
            self.deque.append(state_image)
        stacked_state = np.stack(self.deque, axis=0)
        return stacked_state

    def process_state_for_network(self, state) -> np.ndarray:
        """
        Scale, convert to greyscale and store as float32.
        Basically same as process state for memory, but this time
        outputs float32 images.
        :param state: state in memory-format
        :return: state in network-format
        """
        state = state.astype('float')
        if self.standardize_image:
            state -= 128.0
            state /= 255.0
        return state

    @staticmethod
    def process_reward(reward, reward_clipping=False) -> float:
        """
        Optionally clips reward to {-1, 1}
        :param reward: reward as float
        :param reward_clipping: optional flag, true if reward should be clipped
        :return: processed reward
        """
        if reward_clipping:
            return np.sign(reward)
        return reward
