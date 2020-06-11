from CTL.tree import *
from abc import ABC, abstractmethod


class CTNode(ABC):

    def __init__(self):
        super().__init__()


class CausalTree(ABC):

    def __init__(self):
        super().__init__()

        self.obj = 0.0

        # Haven't implemented "mse" yet
        self.mse = 0.0
        self.tree_depth = 0
        self.num_leaves = 0

    @abstractmethod
    def fit(self, x, y, t):
        pass

    @abstractmethod
    def predict(self, x):
        pass
