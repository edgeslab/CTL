from abc import ABC, abstractmethod


class Node(ABC):

    def __init__(self):
        self.is_leaf = False


class Tree(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, x, y, t):
        pass

    @abstractmethod
    def predict(self, x):
        pass
