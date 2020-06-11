from CTL.causal_tree.nn_pehe.tree import *
from sklearn.model_selection import train_test_split


class BaseNode(PEHENode):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.obj = obj


# ----------------------------------------------------------------
# Base causal tree (ctl, base objective)
# ----------------------------------------------------------------
class BasePEHE(PEHETree):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root = BaseNode()

    def fit(self, x, y, t):
        if x.shape[0] == 0:
            return 0

        # ----------------------------------------------------------------
        # Seed
        # ----------------------------------------------------------------
        np.random.seed(self.seed)

        # ----------------------------------------------------------------
        # Verbosity?
        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        # Split data
        # ----------------------------------------------------------------
        train_x, val_x, train_y, val_y, train_t, val_t = train_test_split(x, y, t, random_state=self.seed, shuffle=True,
                                                                          test_size=self.val_split)
        self.root.num_samples = y.shape[0]
        # ----------------------------------------------------------------
        # effect and pvals
        # ----------------------------------------------------------------
        effect = tau_squared(y, t)
        p_val = get_pval(y, t)
        self.root.effect = effect
        self.root.p_val = p_val

        # ----------------------------------------------------------------
        # Not sure if i should eval in root or not
        # ----------------------------------------------------------------
        node_eval, mse = self._eval(train_y, train_t, val_y, val_t)
        self.root.obj = node_eval

        # ----------------------------------------------------------------
        # Add control/treatment means
        # ----------------------------------------------------------------
        self.root.control_mean = np.mean(y[t == 0])
        self.root.treatment_mean = np.mean(y[t == 1])

        self.root.num_samples = x.shape[0]

        self._fit(self.root, train_x, train_y, train_t, val_x, val_y, val_t)

    def _fit(self, node: BaseNode, train_x, train_y, train_t, val_x, val_y, val_t):

        if train_x.shape[0] == 0 or val_x.shape[0] == 0:
            return node

        if node.node_depth > self.tree_depth:
            self.tree_depth = node.node_depth

        if self.max_depth == self.tree_depth:
            self.num_leaves += 1
            node.leaf_num = self.num_leaves
            node.is_leaf = True
            return node

        best_gain = 0.0
        best_attributes = []
        best_tb_obj, best_fb_obj = (0.0, 0.0)

        column_count = train_x.shape[1]
        for col in range(0, column_count):
            unique_vals = np.unique(train_x[:, col])

            if self.max_values is not None:
                if self.max_values < 1:
                    idx = np.round(np.linspace(
                        0, len(unique_vals) - 1, self.max_values * len(unique_vals))).astype(int)
                    unique_vals = unique_vals[idx]
                else:
                    idx = np.round(np.linspace(
                        0, len(unique_vals) - 1, self.max_values)).astype(int)
                    unique_vals = unique_vals[idx]

            # using the faster evaluation with vector/matrix calculations
            try:
                if self.feature_batch_size is None:
                    split_obj, upper_obj, lower_obj, value = self._eval_fast(train_x, train_y, train_t, val_x, val_y,
                                                                             val_t,
                                                                             unique_vals, col)
                    gain = -node.obj + split_obj
                    if gain > best_gain:
                        best_gain = gain
                        best_attributes = [col, value]
                        best_tb_obj, best_fb_obj = (upper_obj, lower_obj)
                else:

                    for x in batch(unique_vals, self.feature_batch_size):
                        split_obj, upper_obj, lower_obj, value = self._eval_fast(train_x, train_y, train_t, val_x,
                                                                                 val_y, val_t, x, col)

                        gain = -node.obj + split_obj
                        if gain > best_gain:
                            best_gain = gain
                            best_attributes = [col, value]
                            best_tb_obj, best_fb_obj = (upper_obj, lower_obj)
            # if that fails (due to memory maybe?) then use the old calculation
            except:
                for value in unique_vals:

                    (val_x1, val_x2, val_y1, val_y2, val_t1, val_t2) \
                        = divide_set(val_x, val_y, val_t, col, value)

                    # check validation set size
                    val_size = self.val_split * self.min_size if self.val_split * self.min_size > 2 else 2
                    if check_min_size(val_size, val_t1) or check_min_size(val_size, val_t2):
                        continue

                    # check training data size
                    (train_x1, train_x2, train_y1, train_y2, train_t1, train_t2) \
                        = divide_set(train_x, train_y, train_t, col, value)
                    check1 = check_min_size(self.min_size, train_t1)
                    check2 = check_min_size(self.min_size, train_t2)
                    if check1 or check2:
                        continue

                    tb_eval, tb_mse = self._eval(train_y1, train_t1, val_y1, val_t1)
                    fb_eval, fb_mse = self._eval(train_y2, train_t2, val_y2, val_t2)

                    split_eval = (tb_eval + fb_eval)
                    gain = -node.obj + split_eval

                    if gain > best_gain:
                        best_gain = gain
                        best_attributes = [col, value]
                        best_tb_obj, best_fb_obj = (tb_eval, fb_eval)

        if best_gain > 0:
            node.col = best_attributes[0]
            node.value = best_attributes[1]

            # print(node.col)

            (train_x1, train_x2, train_y1, train_y2, train_t1, train_t2) \
                = divide_set(train_x, train_y, train_t, node.col, node.value)

            (val_x1, val_x2, val_y1, val_y2, val_t1, val_t2) \
                = divide_set(val_x, val_y, val_t, node.col, node.value)

            y1 = np.concatenate((train_y1, val_y1))
            y2 = np.concatenate((train_y2, val_y2))
            t1 = np.concatenate((train_t1, val_t1))
            t2 = np.concatenate((train_t2, val_t2))

            best_tb_effect = ace(y1, t1)
            best_fb_effect = ace(y2, t2)
            tb_p_val = get_pval(y1, t1)
            fb_p_val = get_pval(y2, t2)

            self.obj = self.obj - node.obj + best_tb_obj + best_fb_obj

            # ----------------------------------------------------------------
            # Ignore "mse" here, come back to it later?
            # ----------------------------------------------------------------

            tb = BaseNode(obj=best_tb_obj, effect=best_tb_effect, p_val=tb_p_val,
                          node_depth=node.node_depth + 1,
                          num_samples=y1.shape[0])
            fb = BaseNode(obj=best_fb_obj, effect=best_fb_effect, p_val=fb_p_val,
                          node_depth=node.node_depth + 1,
                          num_samples=y2.shape[0])

            node.true_branch = self._fit(tb, train_x1, train_y1, train_t1, val_x1, val_y1, val_t1)
            node.false_branch = self._fit(fb, train_x2, train_y2, train_t2, val_x2, val_y2, val_t2)

            if node.effect > self.max_effect:
                self.max_effect = node.effect
            else:
                self.min_effect = node.effect

            return node

        else:
            if node.effect > self.max_effect:
                self.max_effect = node.effect
            if node.effect < self.min_effect:
                self.min_effect = node.effect

            self.num_leaves += 1
            node.leaf_num = self.num_leaves
            node.is_leaf = True
            return node
