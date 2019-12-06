from utils.preprocessing.Abstract_Preprocess import AbstractProcessor


class Processor(AbstractProcessor):
    def __init__(self):
        super().__init__()
        self.step_max = 500
        self.time_max = 5
        self.reward_min = 0      
        self.reward_max = 500

    def get_state_space(self):
        return " No shape"

    def process_state_for_memory(self, state, is_new_episode):
        return state

    def process_state_for_network(self, state):
        return state
