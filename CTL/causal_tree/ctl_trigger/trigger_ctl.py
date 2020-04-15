from CTL.causal_tree.ctl.binary_ctl import *
# from CTL.causal_tree.util import *
import numpy as np


class TriggerNode(CausalTreeLearnNode):

    def __init__(self, trigger=0.0, **kwargs):
        # ----------------------------------------------------------------
        # Causal tree node
        # ----------------------------------------------------------------
        super().__init__(**kwargs)

        self.trigger = trigger


class TriggerTree(CausalTreeLearn):

    def __init__(self, quartile=True, old_trigger_code=False, **kwargs):
        super().__init__(**kwargs)

        self.quartile = quartile
        self.old_trigger_code = old_trigger_code

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

        if self.old_trigger_code:
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
        else:
            train_effect, best_trigger = tau_squared_trigger(train_y, train_t, self.min_size, self.quartile)
            if train_effect <= -np.inf:
                return return_val

            val_effect = ace_trigger(val_y, val_t, best_trigger)
            if val_effect <= -np.inf:
                return return_val
            train_err = train_effect ** 2

            # train_mse = (1 - self.weight) * (total_train * train_err)
            # cost = self.weight * total_val * np.abs(train_effect - val_effect)
            # # print(train_mse, cost, total_train, total_val, np.abs(total_train - total_val) + 1)
            # obj = (train_mse - cost) / (np.abs(total_train - total_val) + 1)

            train_mse = (1 - self.weight) * (train_effect ** 2)
            cost = self.weight * np.abs(train_effect - val_effect)
            obj = train_mse - cost
            obj = total_train * obj

            best_obj = obj
            # best_mse = train_err
            best_mse = total_train * train_err

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

    # Not working as well as I hoped
    def new_trigger_split(self, train_x, train_y, train_t, val_x, val_y, val_t, unique_vals, col):
        # TODO comments and min_size and val_size
        val_size = self.val_split * self.min_size if self.val_split * self.min_size > 2 else 2

        train_col_x = train_x[:, col]
        val_col_x = val_x[:, col]

        # tile the features, outcome and treatment by the number of potential x-splits
        xx = np.tile(train_col_x, (unique_vals.shape[0], 1))
        yy = np.tile(train_y, (unique_vals.shape[0], 1))
        tt = np.tile(train_t, (unique_vals.shape[0], 1))

        # tile the validation the same way
        val_xx = np.tile(val_col_x, (unique_vals.shape[0], 1))
        val_yy = np.tile(val_y, (unique_vals.shape[0], 1))
        val_tt = np.tile(val_t, (unique_vals.shape[0], 1))

        # find all split indices
        idx_x = np.transpose(np.transpose(xx) >= unique_vals)
        val_idx_x = np.transpose(np.transpose(val_xx) >= unique_vals)

        # get all treatment values
        unique_treatment = np.unique(train_t)
        unique_treatment = (unique_treatment[1:] + unique_treatment[:-1]) / 2
        unique_treatment = unique_treatment[1:-1]

        # tile the treatment by the number of unique_treatments
        ttt = np.tile(tt, (unique_treatment.shape[0], 1, 1))
        final_compare = np.transpose(np.transpose(ttt) >= unique_treatment)
        ttt[final_compare] = 1
        ttt[np.logical_not(final_compare)] = 0

        # tile the validation
        val_ttt = np.tile(val_tt, (unique_treatment.shape[0], 1, 1))
        final_compare_val = np.transpose(np.transpose(val_ttt) >= unique_treatment)
        val_ttt[final_compare_val] = 1
        val_ttt[np.logical_not(final_compare_val)] = 0

        # tile the indices (each "row" is a unique treatment which contains a matrix of (num_vals, num_examples)
        yyy = np.tile(yy, (unique_treatment.shape[0], 1, 1))
        idx_xx = np.tile(idx_x, (unique_treatment.shape[0], 1, 1))

        val_yyy = np.tile(val_yy, (unique_treatment.shape[0], 1, 1))
        val_idx_xx = np.tile(val_idx_x, (unique_treatment.shape[0], 1, 1))

        # counting the number of treated above and below the split, and above and below the trigger
        train_denom_treated_upper = idx_xx * (ttt == 1)
        train_denom_control_upper = idx_xx * (ttt == 0)
        train_treat_nums_upper = np.sum(train_denom_treated_upper, axis=-1) + 1
        train_control_nums_upper = np.sum(train_denom_control_upper, axis=-1) + 1
        train_denom_treated_lower = ~idx_xx * (ttt == 1)
        train_denom_control_lower = ~idx_xx * (ttt == 0)
        train_treat_nums_lower = np.sum(train_denom_treated_lower, axis=-1) + 1
        train_control_nums_lower = np.sum(train_denom_control_lower, axis=-1) + 1

        val_denom_treated_upper = val_idx_xx * (val_ttt == 1)
        val_denom_control_upper = val_idx_xx * (val_ttt == 0)
        val_treat_nums_upper = np.sum(val_denom_treated_upper, axis=-1) + 1
        val_control_nums_upper = np.sum(val_denom_control_upper, axis=-1) + 1
        val_denom_treated_lower = ~val_idx_xx * (val_ttt == 1)
        val_denom_control_lower = ~val_idx_xx * (val_ttt == 0)
        val_treat_nums_lower = np.sum(val_denom_treated_lower, axis=-1) + 1
        val_control_nums_lower = np.sum(val_denom_control_lower, axis=-1) + 1

        return_val = (-np.inf, -np.inf, -np.inf, -np.inf, -np.inf)

        # if train_treat_nums_upper.shape[0] < 1 or train_control_nums_upper.shape[0] < 1 or \
        #         train_treat_nums_lower.shape[0] < 1 or train_control_nums_lower.shape[0] < 1:
        #     return return_val
        #
        # if val_treat_nums_upper.shape[0] < 1 or val_control_nums_upper.shape[0] < 1 or \
        #         val_treat_nums_lower.shape[0] < 1 or val_control_nums_lower.shape[0] < 1:
        #     return return_val

        split_upper_check = np.logical_and(train_treat_nums_upper >= self.min_size,
                                           train_control_nums_upper >= self.min_size)
        split_lower_check = np.logical_and(train_treat_nums_lower >= self.min_size,
                                           train_control_nums_lower >= self.min_size)
        val_split_upper_check = np.logical_and(val_treat_nums_upper >= val_size,
                                               val_control_nums_upper >= val_size)
        val_split_lower_check = np.logical_and(val_treat_nums_lower >= val_size,
                                               val_control_nums_lower >= val_size)

        train_split_check = np.logical_and(split_upper_check, split_lower_check)
        val_split_check = np.logical_and(val_split_upper_check, val_split_lower_check)
        check = ~np.logical_and(train_split_check, val_split_check)
        min_size_idx = np.where(np.logical_and(train_split_check, val_split_check))[0]

        if len(min_size_idx) < 1:
            return return_val

        train_um1 = np.sum(yyy * train_denom_treated_upper,
                           axis=-1) / train_treat_nums_upper
        train_um0 = np.sum(yyy * train_denom_control_upper,
                           axis=-1) / train_control_nums_upper
        train_upper_effect = train_um1 - train_um0
        train_lm1 = np.sum(yyy * train_denom_treated_lower,
                           axis=-1) / train_treat_nums_lower
        train_lm0 = np.sum(yyy * train_denom_control_lower,
                           axis=-1) / train_control_nums_lower
        train_lower_effect = train_lm1 - train_lm0

        val_um1 = np.sum(val_yyy * val_denom_treated_upper,
                         axis=-1) / val_treat_nums_upper
        val_um0 = np.sum(val_yyy * val_denom_control_upper,
                         axis=-1) / val_control_nums_upper
        val_upper_effect = val_um1 - val_um0
        val_lm1 = np.sum(val_yyy * val_denom_treated_lower,
                         axis=-1) / val_treat_nums_lower
        val_lm0 = np.sum(val_yyy * val_denom_control_lower,
                         axis=-1) / val_control_nums_lower
        val_lower_effect = val_lm1 - val_lm0

        upper_obj = train_upper_effect ** 2
        lower_obj = train_lower_effect ** 2
        upper_obj[check] = -1
        lower_obj[check] = -1
        upper_cost = np.abs(train_upper_effect - val_upper_effect)
        lower_cost = np.abs(train_lower_effect - val_lower_effect)
        upper_obj = upper_obj - upper_cost
        lower_obj = lower_obj - lower_cost

        upper_max = np.max(upper_obj, axis=0)
        lower_max = np.max(lower_obj, axis=0)

        upper_trigger_idx = np.argmax(upper_obj, axis=0)
        lower_trigger_idx = np.argmax(lower_obj, axis=0)

        split_obj = upper_max + lower_max
        best_obj_idx = split_obj.argmax()

        upper_trigger = unique_treatment[upper_trigger_idx[best_obj_idx]]
        lower_trigger = unique_treatment[lower_trigger_idx[best_obj_idx]]
        best_value = unique_vals[best_obj_idx]
        upper_val = upper_max[best_obj_idx]
        lower_val = lower_max[best_obj_idx]

        return upper_val, upper_trigger, lower_val, lower_trigger, best_value
