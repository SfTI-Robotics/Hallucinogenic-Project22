from utils.preprocessing.Abstract_Preprocess import AbstractProcessor


class Processor(AbstractProcessor):
    def __init__(self):
        super().__init__()
        self.step_max = 2200
        self.reward_min = -21
        self.reward_max = 21
