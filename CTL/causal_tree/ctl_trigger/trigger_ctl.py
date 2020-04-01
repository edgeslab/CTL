from CTL.causal_tree.ctl.binary_ctl import *
import numpy as np
from abc import ABC, abstractmethod


class TriggerNode(CausalTreeLearnNode):

    def __init__(self, trigger=0.0, **kwargs):
        # ----------------------------------------------------------------
        # Causal tree node
        # ----------------------------------------------------------------
        super().__init__(**kwargs)

        self.trigger = trigger


class TriggerTree(CausalTreeLearn):

    def __init__(self, quartile=True, **kwargs):
        super().__init__(**kwargs)

        self.quartile = quartile

        self.root = TriggerNode()

    @abstractmethod
    def fit(self, x, y, t):
        pass

    def _eval(self, train_y, train_t, val_y, val_t):
        """Continuous case"""
        total_train = train_y.shape[0]
        total_val = val_y.shape[0]

        return_val = (-np.inf, -np.inf, -np.inf)

        if total_train == 0 or total_val == 0:
            return return_val

        unique_treatment = np.unique(train_t)

        if unique_treatment.shape[0] == 1:
            return return_val

        unique_treatment = (unique_treatment[1:] + unique_treatment[:-1]) / 2
        # ignore the first and last
        unique_treatment = unique_treatment[1:-1]

        if self.quartile:
            first_quartile = int(np.floor(unique_treatment.shape[0] / 4))
            third_quartile = int(np.ceil(3 * unique_treatment.shape[0] / 4))

            unique_treatment = unique_treatment[first_quartile:third_quartile]

        # ----------------------------------------------------------------
        # Max values done later
        # ----------------------------------------------------------------
        # if self.max_values < 1:
        #     idx = np.round(np.linspace(
        #         0, len(unique_treatment) - 1, self.max_values * len(unique_treatment))).astype(int)
        #     unique_treatment = unique_treatment[idx]
        # else:
        #     idx = np.round(np.linspace(
        #         0, len(unique_treatment) - 1, self.max_values)).astype(int)
        #     unique_treatment = unique_treatment[idx]

        yyt = np.tile(train_y, (unique_treatment.shape[0], 1))
        ttt = np.tile(train_t, (unique_treatment.shape[0], 1))
        yyv = np.tile(val_y, (unique_treatment.shape[0], 1))
        ttv = np.tile(val_t, (unique_treatment.shape[0], 1))

        xt = np.transpose(np.transpose(ttt) > unique_treatment)
        ttt[xt] = 1
        ttt[np.logical_not(xt)] = 0
        xv = np.transpose(np.transpose(ttv) > unique_treatment)
        ttv[xv] = 1
        ttv[np.logical_not(xv)] = 0

        # do the min_size check on training set
        treat_num = np.sum(ttt == 1, axis=1)
        cont_num = np.sum(ttt == 0, axis=1)
        min_size_idx = np.where(np.logical_and(
            treat_num >= self.min_size, cont_num >= self.min_size))

        unique_treatment = unique_treatment[min_size_idx]
        ttt = ttt[min_size_idx]
        yyt = yyt[min_size_idx]
        ttv = ttv[min_size_idx]
        yyv = yyv[min_size_idx]
        if ttt.shape[0] == 0:
            return return_val
        if ttv.shape[0] == 0:
            return return_val

        # do the min_size check on validation set
        treat_num = np.sum(ttv == 1, axis=1)
        cont_num = np.sum(ttv == 0, axis=1)
        min_size_idx = np.where(np.logical_and(
            treat_num >= self.min_size, cont_num >= self.min_size))

        unique_treatment = unique_treatment[min_size_idx]
        ttt = ttt[min_size_idx]
        yyt = yyt[min_size_idx]
        ttv = ttv[min_size_idx]
        yyv = yyv[min_size_idx]
        if ttt.shape[0] == 0:
            return return_val
        if ttv.shape[0] == 0:
            return return_val

        y_t_m_t = np.sum((yyt * (ttt == 1)), axis=1) / np.sum(ttt == 1, axis=1)
        y_c_m_t = np.sum((yyt * (ttt == 0)), axis=1) / np.sum(ttt == 0, axis=1)

        y_t_m_v = np.sum((yyv * (ttv == 1)), axis=1) / np.sum(ttv == 1, axis=1)
        y_c_m_v = np.sum((yyv * (ttv == 0)), axis=1) / np.sum(ttv == 0, axis=1)

        train_effect = y_t_m_t - y_c_m_t
        train_err = train_effect ** 2

        val_effect = y_t_m_v - y_c_m_v
        # val_err = val_effect ** 2

        train_mse = (1 - self.weight) * (total_train * train_err)
        cost = self.weight * total_val * np.abs(train_effect - val_effect)
        obj = (train_mse - cost) / (np.abs(total_train - total_val) + 1)

        argmax_obj = np.argmax(obj)
        best_obj = obj[argmax_obj]
        best_trigger = unique_treatment[argmax_obj]
        best_mse = train_err[argmax_obj]

        return best_obj, best_trigger, best_mse

    def get_triggers(self, x):
        def _get_features(node: TriggerNode, observation):
            if node.is_leaf:
                return node.trigger
            else:
                v = observation[node.col]
                if v >= node.value:
                    branch = node.true_branch
                else:
                    branch = node.false_branch

            return _get_features(branch, observation)

        if len(x.shape) == 1:
            return _get_features(self.root, x)
        num_test = x.shape[0]
        triggers = np.zeros(num_test)

        for i in range(num_test):
            test_example = x[i, :]
            triggers[i] = _get_features(self.root, test_example)

        return triggers
