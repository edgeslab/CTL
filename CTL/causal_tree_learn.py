from CTL.causal_tree.ctl.adaptive import *
from CTL.causal_tree.ctl.ctl_base import *
from CTL.causal_tree.ctl.ctl_honest import *
from CTL.causal_tree.ctl.ctl_val_honest import *

from CTL.causal_tree.ctl_trigger.adaptive_trigger import *
from CTL.causal_tree.ctl_trigger.ctl_base_trigger import *
from CTL.causal_tree.ctl_trigger.ctl_honest_trigger import *
from CTL.causal_tree.ctl_trigger.ctl_val_honest_trigger import *


class CausalTree:

    def __init__(self, cont=False, val_honest=False, honest=False, min_size=2, max_depth=-1, split_size=0.5, weight=0.5,
                 seed=724, quartile=False, old_trigger_code=False):
        self.cont = cont
        if cont:
            if split_size <= 0.0 and weight <= 0.0:
                self.tree = AdaptiveTriggerTree(min_size=min_size, max_depth=max_depth,
                                                split_size=split_size, weight=weight, seed=seed,
                                                quartile=quartile, old_trigger_code=old_trigger_code)
            elif val_honest and weight > 0.0:
                self.tree = TriggerTreeHonestValidation(min_size=min_size, max_depth=max_depth,
                                                        split_size=split_size, weight=weight, seed=seed,
                                                        quartile=quartile, old_trigger_code=old_trigger_code)
            elif honest and weight > 0.0:
                self.tree = TriggerTreeHonest(min_size=min_size, max_depth=max_depth, split_size=split_size,
                                              weight=weight, seed=seed, quartile=quartile,
                                              old_trigger_code=old_trigger_code)
            elif weight > 0.0 and split_size > 0.0:
                self.tree = TriggerTreeBase(min_size=min_size, max_depth=max_depth, split_size=split_size,
                                            weight=weight, seed=seed, quartile=quartile,
                                            old_trigger_code=old_trigger_code)
            else:
                self.tree = AdaptiveTriggerTree(min_size=min_size, max_depth=max_depth,
                                                split_size=split_size, weight=weight, seed=seed,
                                                quartile=quartile, old_trigger_code=old_trigger_code)
        else:
            if split_size <= 0.0 and weight <= 0.0:
                self.tree = AdaptiveTree(min_size=min_size, max_depth=max_depth,
                                         split_size=split_size, weight=weight, seed=seed)
            elif val_honest and weight > 0.0:
                self.tree = CausalTreeLearnHonestValidation(min_size=min_size, max_depth=max_depth,
                                                            split_size=split_size, weight=weight, seed=seed)
            elif honest and weight > 0.0:
                self.tree = CausalTreeLearnHonest(min_size=min_size, max_depth=max_depth, split_size=split_size,
                                                  weight=weight, seed=seed)
            elif weight > 0.0 and split_size > 0.0:
                self.tree = CausalTreeLearnBase(min_size=min_size, max_depth=max_depth, split_size=split_size,
                                                weight=weight, seed=seed)
            else:
                self.tree = AdaptiveTree(min_size=min_size, max_depth=max_depth,
                                         split_size=split_size, weight=weight, seed=seed)

        self.column_num = 0
        self.fitted = False
        self.tree_depth = 0

    def fit(self, x, y, t):
        self.column_num = x.shape[1]
        x = x.astype(np.float)
        y = y.astype(np.float)
        t = t.astype(np.float)
        self.tree.fit(x, y, t)
        self.fitted = True
        self.tree_depth = self.tree.tree_depth

    def predict(self, x):
        if self.fitted:
            return self.tree.predict(x)
        else:
            return "Tree not fitted yet!"

    def prune(self, alpha=0.05):
        self.tree.prune(alpha=alpha)

    def get_groups(self, x):
        return self.tree.get_groups(x)

    def get_features(self, x):
        return self.tree.get_features(x)

    def get_triggers(self, x):
        if self.cont:
            return self.tree.get_triggers(x)
        else:
            return "Need to be a trigger tree"

    # ----------------------------------------------------------------
    # Plotting and printing trees
    # ----------------------------------------------------------------
    def plot_tree(self, filename="tree", features=None, training_data=None, alpha=0.05, show_pval=True, dpi=100,
                  show_samples=True,
                  show_effect=True, trigger_precision=2, extension="png", create_png=True):
        if not self.fitted:
            return "Tree not fitted yet!"

        if features is None:
            if self.tree.features is not None:
                feature_names = self.tree.features
            else:
                feature_names = []
                for i in range(self.column_num):
                    feature_names.append(f"att_{i}")

        else:
            feature_names = features

        name_split = filename.split("/")
        if len(name_split) > 1:
            img_folder = name_split[0:-1]
            file_name = name_split[-1]

            img_folder = "/".join(img_folder)

            dot_folder = img_folder + "/dot_folder/"

            check_dir(img_folder + "/")
            check_dir(dot_folder)

            dot_file_name = dot_folder + file_name
            img_file_name = filename
        else:
            dot_file_name = filename
            img_file_name = filename

        self._tree_to_dot(self.tree, feature_names, dot_file_name, alpha=alpha, show_pval=show_pval,
                          show_samples=show_samples, show_effect=show_effect, trigger_precision=trigger_precision)
        if create_png:
            self._dot_to_png(dot_file_name, img_file_name, extension=extension, dpi=dpi)

    @staticmethod
    def _dot_to_png(dot_filename="tree", output_file=None, extension="png", dpi=100):

        if output_file is None:
            command = ["dot", "-T" + extension, f"-Gdpi={dpi}", dot_filename + '.dot', "-o",
                       dot_filename + "." + extension]
        else:
            command = ["dot", "-T" + extension, f"-Gdpi={dpi}",
                       dot_filename + '.dot', "-o", output_file + "." + extension]
        try:
            if os.name == 'nt':
                subprocess.check_call(command, shell=True)
            else:
                subprocess.check_call(command)
        except subprocess.CalledProcessError:
            exit("Could not run dot, ie graphviz, to "
                 "produce visualization")

    def _tree_to_dot(self, tree, features, filename, alpha=0.05, show_pval=False, show_samples=True, show_effect=True,
                     trigger_precision=2):
        filename = filename + ".dot"
        feat_names = col_dict(features)
        with open(filename, "w") as dot_file:
            dot_file.write('digraph Tree {\n')
            dot_file.write('node [shape=box, fontsize=32] ;\n')
            dot_file.write('edge [fontsize=24] ;\n')
            self._tree_to_dot_r(tree.root, feat_names, dot_file, counter=0, alpha=alpha, show_pval=show_pval,
                                show_samples=show_samples, show_effect=show_effect, trigger_precision=trigger_precision)
            dot_file.write("}")

    def _tree_to_dot_r(self, node: CausalTreeLearnNode, features, dot_file, counter, alpha=0.5, show_pval=True,
                       show_samples=True,
                       show_effect=True, trigger_precision=2):

        curr_node = counter
        dot_file.write(str(counter) + ' ')
        dot_file.write('[')
        node_str = list(['label=\"'])

        # add effect
        if show_effect:
            node_str.append('effect = ')
            effect_str = "%.3f" % node.effect
            node_str.append(effect_str)

        # ----------------------------------------------------------------
        # Triggers
        # ----------------------------------------------------------------
        if self.cont:
            node_str.append('\\ntrigger > ')
            treat_str = "{1:.{0}f}".format(trigger_precision, node.trigger)
            node_str.append(treat_str)

        # p_values
        if show_pval:
            node_str.append('\\np = ')
            p_val_str = "%.3f" % node.p_val
            node_str.append(p_val_str)

        # ----------------------------------------------------------------
        # Number of samples
        # ----------------------------------------------------------------
        if show_samples:
            node_str.append('\\nsamples = ')
            node_str.append(str(node.num_samples))

        # ----------------------------------------------------------------
        # Feature split
        # ----------------------------------------------------------------
        if not node.is_leaf:
            sz_col = 'Column %s' % node.col
            if features and sz_col in features:
                sz_col = features[sz_col]
            if isinstance(node.value, int):
                decision = '%s >= %s' % (sz_col, node.value)
                # opp_decision = '%s < %s' % (sz_col, tree.value)
            elif isinstance(node.value, float):
                decision = '%s >= %.3f' % (sz_col, node.value)
                # opp_decision = '%s < %.3f' % (sz_col, tree.value)
            else:
                decision = '%s == %s' % (sz_col, node.value)
                # opp_decision = '%s =/=' % (sz_col, tree.value)
            node.feature_split = decision

            # if curr_node == 0:
            #     node_str.append('Splitting feature: ')
            node_str.append('\\n' + decision + '\\n')

        # ----------------------------------------------------------------
        # The end
        # ----------------------------------------------------------------
        node_str.append('\"')

        # ----------------------------------------------------------------
        # Color fill
        # ----------------------------------------------------------------
        node_str.append(", style=filled")

        color = '\"#ffffff\"'
        color_idx = 0
        effect = node.effect
        eps = 0.01
        if np.abs(effect) <= eps:
            color = "white"
        else:
            if effect > 0:
                # effect_range = np.linspace(0, self.max, 10)
                effect_range = np.linspace(0, 1, 10)
                for idx, effect_r in enumerate(effect_range[:-1]):
                    if effect_range[idx] <= effect <= effect_range[idx + 1]:
                        color = "\"/blues9/%i\"" % (idx + 1)
                        color_idx = idx
                        break
                if color_idx >= 8:
                    font_color = ", fontcolor=white"
                    node_str.append(font_color)
            else:
                # effect_range = np.linspace(self.min, 0, 10)
                effect_range = np.linspace(-1, 0, 10)[::-1]
                for idx, effect_r in enumerate(effect_range[:-1]):
                    # if effect_range[idx] >= effect >= effect_range[idx + 1]:
                    #         color = "\"/reds9/%i\"" % (idx + 1)
                    #         color_idx = idx
                    #         break
                    # if effect <= effect_range[idx] and effect >= effect_range[idx+1]:
                    if effect_range[idx + 1] <= effect <= effect_range[idx]:
                        color = "\"/reds9/%i\"" % (idx + 1)
                        color_idx = idx
                        break
                if color_idx >= 8:
                    font_color = ", fontcolor=white"
                    node_str.append(font_color)

        color_str = ", fillcolor=" + color
        node_str.append(color_str)

        # ----------------------------------------------------------------
        # p-value highlighting
        # ----------------------------------------------------------------

        if node.p_val <= alpha:
            # node_str.append(", shape=box")
            # node_str.append(", sides=4")
            # node_str.append(", peripheries=3")
            node_str.append(", color=purple")
            node_str.append(", penwidth=10.0")

        node_str.append('] ;\n')
        dot_file.write(''.join(node_str))

        # ----------------------------------------------------------------
        # start doing the branches
        # ----------------------------------------------------------------
        counter = counter + 1
        if node.true_branch is not None:
            if curr_node == 0:
                dot_file.write(str(curr_node) + ' -> ' + str(
                    counter) + ' [labeldistance=2.5, labelangle=45, headlabel=\"True\", color=green, penwidth=2] ;\n')
            else:
                dot_file.write(str(curr_node) + ' -> ' + str(counter) + '[color=green, penwidth=2] ;\n')
            # f.write(str(curr_node) + ' -> ' + str(counter) +
            #         ' [labeldistance=2.5, labelangle=45, headlabel=' + decision + '];\n')
            counter = self._tree_to_dot_r(
                node.true_branch, features, dot_file, counter, alpha=alpha, show_pval=show_pval,
                show_samples=show_samples,
                show_effect=show_effect, trigger_precision=trigger_precision)
        if node.false_branch is not None:
            if curr_node == 0:
                dot_file.write(str(curr_node) + ' -> ' + str(
                    counter) + ' [labeldistance=2.5, labelangle=-45, headlabel=\"False\", color=red, penwidth=2] ;\n')
            else:
                dot_file.write(str(curr_node) + ' -> ' + str(counter) + '[color=red, penwidth=2] ;\n')
            # f.write(str(curr_node) + ' -> ' + str(counter) +
            #         ' [labeldistance=2.5, labelangle=45, headlabel=' + opp_decision + '];\n')
            counter = self._tree_to_dot_r(
                node.false_branch, features, dot_file, counter, alpha=alpha, show_pval=show_pval,
                show_samples=show_samples,
                show_effect=show_effect, trigger_precision=trigger_precision)

        return counter

    # ----------------------------------------------------------------
    # Variable names
    # ----------------------------------------------------------------

    def assign_feature_names(self, feature_names):

        self.tree.features = feature_names

        variable_names = col_dict(feature_names)

        def _assign_feature_names(node: CausalTreeLearnNode, feat_names):

            if not node.is_leaf:
                sz_col = 'Column %s' % node.col
                if feat_names and sz_col in feat_names:
                    sz_col = feat_names[sz_col]
                decision = '%s' % sz_col
                node.column_name = decision

                sz_col = 'Column %s' % node.col
                if feat_names and sz_col in feat_names:
                    sz_col = feat_names[sz_col]
                if isinstance(node.value, int):
                    decision = '%s >= %s' % (sz_col, node.value)
                    # opp_decision = '%s < %s' % (sz_col, tree.value)
                elif isinstance(node.value, float):
                    decision = '%s >= %.3f' % (sz_col, node.value)
                    # opp_decision = '%s < %.3f' % (sz_col, tree.value)
                else:
                    decision = '%s == %s' % (sz_col, node.value)
                    # opp_decision = '%s =/=' % (sz_col, tree.value)
                node.decision = decision

            # start doing the branches
            if node.true_branch is not None:
                _assign_feature_names(node.true_branch, feat_names)
            if node.false_branch is not None:
                _assign_feature_names(node.false_branch, feat_names)

        _assign_feature_names(self.tree.root, variable_names)

    def get_features_used(self, variable_names=None, cat=False):
        return self.get_variables_used(variable_names, cat)

    def get_variables_used(self, variable_names=None, cat=False):

        if self.tree.features is None:
            if variable_names is not None:
                self.assign_feature_names(feature_names=variable_names)

        def _get_variables(node: CausalTreeLearnNode, list_vars, list_depths):

            # print(node.is_leaf, node.true_branch, node.false_branch)
            if node.is_leaf:
                return list_vars, list_depths
            else:
                if cat:
                    if "==" in node.decision:
                        list_fs = node.decision.split("==")
                        list_fs = [i.strip() for i in list_fs]
                        to_append = "_".join(list_fs)
                        if to_append not in list_vars:
                            list_vars.append(to_append)
                            list_depths.append(node.node_depth)
                    else:
                        if node.decision not in list_vars:
                            list_vars.append(node.column_name)
                            list_depths.append(node.node_depth)
                else:
                    if node.column_name not in list_vars:
                        list_vars.append(node.column_name)
                        list_depths.append(node.node_depth)

                list_vars, list_depths = _get_variables(node.true_branch, list_vars, list_depths)
                list_vars, list_depths = _get_variables(node.false_branch, list_vars, list_depths)

                return list_vars, list_depths

        list_of_vars = []
        list_of_depths = []
        list_of_vars, list_of_depths = _get_variables(self.tree.root, list_of_vars, list_of_depths)

        sorted_vars = []
        sorted_idx = np.argsort(list_of_depths)
        for i in sorted_idx:
            sorted_vars.append(list_of_vars[i])

        return sorted_vars
