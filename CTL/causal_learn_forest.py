from CTL.causal_tree_learn import CausalTree
import numpy as np


class CausalTreeLearnForest:

    def __init__(self, num_trees=10, bootstrap=True, max_samples=None, max_features="auto", max_depth=-1,
                 val_honest=False, honest=False, min_size=2, split_size=0.5, weight=0.5, feature_batch_size=None,
                 seed=724):

        tree_params = {
            "weight": weight,
            "split_size": split_size,
            "max_depth": max_depth,
            "seed": seed,
            "min_size": min_size,
            "val_honest": val_honest,
            "honest": honest,
            "feature_batch_size": feature_batch_size,
        }

        self.num_trees = num_trees
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.max_features = max_features
        self.max_depth = max_depth

        self.trees = tuple(CausalTree(**tree_params) for i in range(num_trees))

    def fit(self, x, y, t):
        x = x.astype(float)
        y = y.astype(float)
        t = t.astype(float)

        for tree in self.trees:
            example_samples, feature_samples = self._sample(x)

            sample_x = x[np.ix_(example_samples, feature_samples)]
            sample_y = y[example_samples]
            sample_t = t[example_samples]

            tree.fit(sample_x, sample_y, sample_t)

    def predict(self, x):
        predictions = np.zeros((self.num_trees, x.shape[0]))
        for i, tree in enumerate(self.trees):
            predictions[i] = tree.predict(x)

        return np.mean(predictions, axis=0)

    def _sample(self, x):
        total_examples = x.shape[0]
        total_features = x.shape[1]

        example_samples = self._sample_examples(total_examples)
        feature_samples = self._feature_sample(total_features)

        return example_samples, feature_samples

    def _sample_examples(self, total_examples):
        if self.bootstrap:
            if self.max_samples:
                if isinstance(self.max_samples, float):
                    example_samples = np.random.choice(np.arange(0, total_examples),
                                                       size=int(self.max_samples * total_examples))
                elif isinstance(self.max_samples, int):
                    example_samples = np.random.choice(np.arange(0, total_examples), size=self.max_samples)
                else:
                    example_samples = np.random.choice(np.arange(0, total_examples), size=total_examples)
            else:
                example_samples = np.random.choice(np.arange(0, total_examples), size=total_examples)
        else:
            example_samples = np.arange(0, total_examples)

        return example_samples

    def _feature_sample(self, total_features):
        num_features = self._feature_sample_size(total_features)
        feature_samples = np.random.permutation(total_features)[:num_features]
        return feature_samples

    def _feature_sample_size(self, total_features):
        num_features = total_features
        if self.max_features == "auto" or self.max_features == "sqrt":
            num_features = int(np.sqrt(num_features))
        elif isinstance(self.max_features, int):
            num_features = self.max_features
        elif isinstance(self.max_features, float):
            num_features = int(self.max_features * total_features)
        return num_features
