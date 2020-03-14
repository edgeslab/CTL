from CTL.ct_util import *
import numpy as np
from abc import ABC, abstractmethod


class CausalTree(ABC):

    @abstractmethod
    def fit(self, x, y, t):
        pass

    @abstractmethod
    def fit_r(self, train_x, train_y, train_t, val_x, val_y, val_t):
        pass

    @abstractmethod
    def predict(self):
        pass