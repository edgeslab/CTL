from abc import ABC, abstractmethod


class Node(ABC):

    def __init__(self):
        pass


class Tree(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, x, y, t):
        pass

    @abstractmethod
    def predict(self, x):
        pass
