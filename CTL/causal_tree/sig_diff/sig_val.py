from CTL.causal_tree.sig_diff.sig import *
from sklearn.model_selection import train_test_split


class SigValNode(SigNode):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SigTreeVal(SigTree):

    def __init__(self, split_size=0.5, **kwargs):
        super().__init__(**kwargs)
        self.split_size = 0.5
        self.root = SigValNode()
        if self.split_size < 0.5:
            self.val_size = self.min_size * self.split_size if self.split_size * self.min_size > 2 else 2
        else:
            self.val_size = self.min_size
        # self.val_size = self.val_split * self.min_size if self.val_split * self.min_size > 2 else 2

    def fit(self, x, y, t):
        if x.shape[0] == 0:
            return 0

        # ----------------------------------------------------------------
        # Seed
        # ----------------------------------------------------------------
        np.random.seed(self.seed)

        # ----------------------------------------------------------------
        # Split data
        # ----------------------------------------------------------------
        train_x, val_x, train_y, val_y, train_t, val_t = train_test_split(x, y, t, random_state=self.seed, shuffle=True,
                                                                          test_size=self.split_size)
        self.root.num_samples = train_y.shape[0]
        # ----------------------------------------------------------------
        # effect and pvals
        # ----------------------------------------------------------------
        effect = tau_squared(y, t)
        p_val = get_pval(y, t)
        self.root.effect = effect
        self.root.p_val = p_val

        self.root.obj = 0
        # ----------------------------------------------------------------
        # Add control/treatment means
        # ----------------------------------------------------------------
        self.root.control_mean = np.mean(y[t == 0])
        self.root.treatment_mean = np.mean(y[t == 1])

        self.root.num_samples = x.shape[0]

        self._fit(self.root, train_x, train_y, train_t, val_x, val_y, val_t)

    def _eval_val(self, train_y1, train_t1, train_y2, train_t2, val_y1, val_t1, val_y2, val_t2):

        total1 = train_y1.shape[0]
        total2 = train_y2.shape[0]
        val_total1 = val_y1.shape[0]
        val_total2 = val_y2.shape[0]

        return_val = (1, 1)
        if total1 < 1 or total2 < 1 or val_total1 < 1 or val_total2 < 1:
            return return_val

        effect1, std1 = self._eval_util(train_y1, train_t1)
        effect2, std2 = self._eval_util(train_y2, train_t2)
        val_effect1, val_std1 = self._eval_util(val_y1, val_t1)
        val_effect2, val_std2 = self._eval_util(val_y2, val_t2)

        training_stat, training_pval = ttest_ind_from_stats(effect1, std1, total1, effect2, std2, total2)
        val_stat, val_pval = ttest_ind_from_stats(val_effect1, val_std1, val_total1, val_effect2, val_std2, val_total2)

        train_val_same_stat1, train_val_same_pval1 = ttest_ind_from_stats(effect1, std1, total1, val_effect1, val_std1,
                                                                          val_total1)
        train_val_same_stat2, train_val_same_pval2 = ttest_ind_from_stats(effect2, std2, total2, val_effect2, val_std2,
                                                                          val_total2)

        train_val_diff_stat1, train_val_diff_pval1 = ttest_ind_from_stats(effect2, std2, total2, val_effect1, val_std1,
                                                                          val_total1)
        train_val_diff_stat2, train_val_diff_pval2 = ttest_ind_from_stats(effect1, std1, total1, val_effect2, val_std2,
                                                                          val_total2)

        eval_stats = {
            "training_stat": training_stat,
            "val_stat": val_stat,
            "train_val_same_stat1": train_val_same_stat1,
            "train_val_same_stat2": train_val_same_stat2,
            "train_val_diff_stat1": train_val_diff_stat1,
            "train_val_diff_stat2": train_val_diff_stat2,
        }

        eval_pvals = {
            "training_pval": training_pval,
            "val_pval": val_pval,
            "train_val_same_pval1": train_val_same_pval1,
            "train_val_same_pval2": train_val_same_pval2,
            "train_val_diff_pval1": train_val_diff_pval1,
            "train_val_diff_pval2": train_val_diff_pval2,
        }

        return eval_stats, eval_pvals

    def _eval_pvals(self, eval_pvals):

        train_check = eval_pvals["training_pval"] <= self.alpha
        val_check = eval_pvals["val_pval"] <= self.alpha

        train_same_check = eval_pvals["train_val_same_pval1"] > self.alpha and eval_pvals[
            "train_val_same_pval2"] > self.alpha

        val_diff_check = eval_pvals["train_val_diff_pval1"] <= self.alpha and eval_pvals[
            "train_val_diff_pval2"] <= self.alpha

        result = train_check and train_same_check and val_check and val_diff_check

        return result

    def _pval_scores(self, eval_pvals):
        training_pval = eval_pvals["training_pval"]
        val_pval = eval_pvals["val_pval"]
        train_val_same_pval1 = eval_pvals["train_val_same_pval1"]
        train_val_same_pval2 = eval_pvals["train_val_same_pval2"]
        train_val_dff_pval1 = eval_pvals["train_val_dff_pval1"]
        train_val_diff_pval2 = eval_pvals["train_val_diff_pval2"]

    def _fit(self, node: SigValNode, train_x, train_y, train_t, val_x, val_y, val_t):

        if train_x.shape[0] == 0:
            return node

        if node.node_depth > self.tree_depth:
            self.tree_depth = node.node_depth

        if self.max_depth == self.tree_depth:
            if node.effect > self.max_effect:
                self.max_effect = node.effect
            if node.effect < self.min_effect:
                self.min_effect = node.effect
            self.num_leaves += 1
            node.leaf_num = self.num_leaves
            node.is_leaf = True
            return node

        best_gain = 1.0
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

            for value in unique_vals:

                (val_x1, val_x2, val_y1, val_y2, val_t1, val_t2) \
                    = divide_set(val_x, val_y, val_t, col, value)
                if check_min_size(self.val_size, val_t1) or check_min_size(self.val_size, val_t2):
                    continue

                # check training data size
                (train_x1, train_x2, train_y1, train_y2, train_t1, train_t2) \
                    = divide_set(train_x, train_y, train_t, col, value)
                check1 = check_min_size(self.min_size, train_t1)
                check2 = check_min_size(self.min_size, train_t2)
                if check1 or check2:
                    continue

                eval_stats, eval_pvals = self._eval_val(train_y1, train_t1, train_y2, train_t2,
                                                        val_y1, val_t1, val_y2, val_t2)

                result = self._eval_pvals(eval_pvals)

                gain = eval_pvals["training_pval"]

                if gain < best_gain and result:
                    best_gain = gain
                    best_attributes = [col, value]

        if best_gain <= self.alpha:
            node.col = best_attributes[0]
            node.value = best_attributes[1]

            (train_x1, train_x2, train_y1, train_y2, train_t1, train_t2) \
                = divide_set(train_x, train_y, train_t, node.col, node.value)

            (val_x1, val_x2, val_y1, val_y2, val_t1, val_t2) \
                = divide_set(val_x, val_y, val_t, node.col, node.value)

            y1 = train_y1
            y2 = train_y2
            t1 = train_t1
            t2 = train_t2

            best_tb_effect = ace(y1, t1)
            best_fb_effect = ace(y2, t2)
            tb_p_val = get_pval(y1, t1)
            fb_p_val = get_pval(y2, t2)

            self.obj = self.obj - node.obj + best_tb_obj + best_fb_obj

            tb = SigValNode(obj=best_tb_obj, effect=best_tb_effect, p_val=tb_p_val,
                            node_depth=node.node_depth + 1,
                            num_samples=y1.shape[0])
            fb = SigValNode(obj=best_fb_obj, effect=best_fb_effect, p_val=fb_p_val,
                            node_depth=node.node_depth + 1,
                            num_samples=y2.shape[0])

            node.true_branch = self._fit(tb, train_x1, train_y1, train_t1, val_x1, val_y1, val_t1)
            node.false_branch = self._fit(fb, train_x2, train_y2, train_t2, val_x2, val_y2, val_t2)

            if node.effect > self.max_effect:
                self.max_effect = node.effect
            if node.effect < self.min_effect:
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
