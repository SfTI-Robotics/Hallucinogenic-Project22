from utils.preprocessing.Abstract_Preprocess import AbstractProcessor


class Processor(AbstractProcessor):

    def __init__(self):
        super().__init__()
        self.step_max = 2000
        self.reward_min = 0
        self.reward_max = 1000
