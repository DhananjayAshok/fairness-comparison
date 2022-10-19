import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class Evaluator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.trajectories = {}

    def track_metric(self, name, metric_fn, model_name=None):
        """
        Meant to store
        :param name:
        :param model_name: should not be 'metric'
        :param metric_fn: takes in y_true, y_pre
        :return:
        """
        if name in self.trajectories:
            assert model_name is not None
            self.trajectories[name][model_name] = []
            assert metric_fn == self.trajectories[name]['metric']
        else:
            if model_name is None:
                self.trajectories[name] = {"metric": metric_fn}
            else:
                self.trajectories[name] = {"metric": metric_fn, model_name: []}

    def update_sklearn_metric(self, y_true, y_pred, name, model_name, k=None):
        """

        :param y_true:
        :param y_pred:
        :param name:
        :param model_name:
        :param k: if k not None then we assume we are doing a CV with folds, then we track a list for every entry
            then the trajectories is of form: trajectories[name][model_name] = [[e_1f_1, e_1f_2], ....[e_nf_1, e_nf_2]]
        :return:
        """
        assert name in self.trajectories
        val = self.trajectories[name]['metric'](y_true, y_pred)
        self.direct_update_metric(val, name, model_name, k=k)

    def update_fairness_metric(self, X, y, pred, name, model_name, k=None, model=None):
        assert name in self.trajectories
        val = self.trajectories[name]['metric'](self.dataset.protected_col, X, y, pred, model=model)
        self.direct_update_metric(val, name, model_name, k=k)

    def direct_update_metric(self, val, name, model_name, k=None):
        assert name in self.trajectories
        assert model_name in self.trajectories[name]
        if k is None:
            self.trajectories[name][model_name].append(val)
        else:
            n_epochs_so_far = len(self.trajectories[name][model_name])
            if k < n_epochs_so_far:
                self.trajectories[name][model_name][k].append(val)
            else:
                # Then this is the first fold and we need to add the list with value.
                self.trajectories[name][model_name].append([val])
        return

    def summarize_metric(self, name):
        assert name in self.trajectories
        traj = self.trajectories[name]
        traj.pop('metric')
        data = []
        columns = ["Fold", "Epoch", name, "Model"]
        model_0 = list(traj.keys())[0]
        els = traj[model_0]
        if type(els[0]) == list:
            for model_name in traj.keys():
                epochs = traj[model_name]
                for epoch in range(len(epochs)):
                    for fold in range(len(epochs[epoch])):
                        fold_res = epochs[epoch][fold]
                        data.append([fold, epoch, fold_res, model_name])
        else:  # Then its numeric values
            for model_name in traj.keys():
                epochs = traj[model_name]
                for epoch in range(epochs):
                    data.append([0, epoch, epochs[epoch], model_name])
        df = pd.DataFrame(data=data, columns=columns)
        self.trajectories[name] = df

    def summarize_metrics(self):
        for name in self.trajectories:
            self.summarize_metric(name)

    def select_model_subset_df(self, name, model_subset):
        assert name in self.trajectories
        df = self.trajectories[name]
        if model_subset is None:
            model_subset = df["Model"].unique().tolist()
        else:
            model_subset = list(df["Model"].unique().toset().intersect(set(model_subset)))
        assert len(model_subset) > 0
        df = df[df["Model"].isin(model_subset)]
        return df

    def select_model_subset_df_multi(self, names, model_subset):
        """
        Assumes all have same epochs, folds, models etc.
        :param names:
        :param model_subset:
        :return:
        """
        assert all([name in self.trajectories for name in names])
        df = self.trajectories[names[0]]
        if model_subset is None:
            model_subset = df["Model"].unique().tolist()
        else:
            model_subset = list(df["Model"].unique().toset().intersect(set(model_subset)))
        assert len(model_subset) > 0
        for name in names[1:]:
            df[name] = self.trajectories[name][name]
        df = df[df["Model"].isin(model_subset)]
        return df

    def plot_metric_trajectory(self, name, model_subset=None):
        assert name in self.trajectories and type(self.trajectories[name]) == pd.DataFrame
        df = self.select_model_subset_df(name, model_subset)
        sns.lineplot(data=df, x="Epoch", y=name, hue="Model")
        plt.show()
        return

    def plot_metric_final(self, name, model_subset=None):
        assert name in self.trajectories and type(self.trajectories[name]) == pd.DataFrame
        df = self.select_model_subset_df(name, model_subset)
        last_epoch = df["Epoch"].max()
        last_only = df[df["Epoch"] == last_epoch]
        sns.boxplot(data=last_only, x="Model", y=name)
        plt.show()
        return

    def plot_metrics_final(self, name_0, name_1, model_subset=None):
        assert [name in self.trajectories and type(self.trajectories[name]) == pd.DataFrame for name in
                [name_0, name_1]]
        df = self.select_model_subset_df_multi([name_0, name_1], model_subset=model_subset)
        last_epoch = df["Epoch"].max()
        last_only = df[df["Epoch"] == last_epoch]
        sns.scatterplot(data=last_only, x=name_0, y=name_1, hue="Model", style="Model")
        plt.show()


class FairnessMetrics:
    def __int__(self):
        pass

    @staticmethod
    def safe_div(x, y):
        if y == 0:
            return 0
        else:
            return x / y

    @staticmethod
    def get_comparator(key):
        if key in ["ratio", "div"]:
            return FairnessMetrics.safe_div
        elif key == "diff":
            return lambda x, y: x-y
        else:
            raise ValueError

    @staticmethod
    def get_aggregator(key):
        if key == "sum":
            return sum
        if key == "max":
            return max
        if key == "min":
            return min
        if key == "mean":
            return lambda x: sum(x)/len(x)

    @staticmethod
    def positive_probability(X, y, pred):
        return pred.mean()

    @staticmethod
    def recall(X, y, pred):
        return metrics.recall_score(y, pred)

    @staticmethod
    def precision(X, y, pred):
        return metrics.precision_score(y, pred)

    @staticmethod
    def tnr(X, y, pred):
        cm = metrics.confusion_matrix(y, pred)
        assert cm.shape == (2,2)
        if cm[1, 0] + cm[1, 1] == 0:
            return 0
        else:
            return cm[1, 1]/(cm[1, 0] + cm[1, 1])

    @staticmethod
    def accuracy(X, y, pred):
        return metrics.accuracy_score(y, pred)

    @staticmethod
    def group_metric(eval_fn, protected_col, X, y, pred, comparator="div", aggregator="max"):
        comparator = FairnessMetrics.get_comparator(comparator)
        aggregator = FairnessMetrics.get_aggregator(aggregator)
        group_values = []
        for group in X[protected_col].unique():
            grp_idx = X[protected_col] == group
            if not grp_idx.any():
                group_values.append(0)
            X_slice = X[grp_idx]
            y_slice = y[grp_idx]
            pred_slice = pred[grp_idx]
            group_value = eval_fn(X_slice, y_slice, pred_slice)
            group_values.append(group_value)
        group_comparisons = []
        for p in group_values:
            for q in group_values:
                if comparator == "div":
                    if q == 0:
                        continue
                group_comparisons.append(comparator(p, q))
        return aggregator(group_comparisons)

    @staticmethod
    def positive_parity(protected_col, X, y, pred, comparator="div", aggregator="max", model=None):
        return FairnessMetrics.group_metric(FairnessMetrics.positive_probability, protected_col, X, y, pred, comparator=comparator,
                                            aggregator=aggregator)

    @staticmethod
    def recall_parity(protected_col, X, y, pred, comparator="div", aggregator="max", model=None):
        return FairnessMetrics.group_metric(FairnessMetrics.recall, protected_col, X, y, pred, comparator=comparator,
                                            aggregator=aggregator)

    @staticmethod
    def accuracy_parity(protected_col, X, y, pred, comparator="diff", aggregator="max", model=None):
        return FairnessMetrics.group_metric(FairnessMetrics.accuracy, protected_col, X, y, pred, comparator=comparator,
                                            aggregator=aggregator)

    @staticmethod
    def precision_parity(protected_col, X, y, pred, comparator="diff", aggregator="max", model=None):
        return FairnessMetrics.group_metric(FairnessMetrics.precision, protected_col, X, y, pred, comparator=comparator,
                                            aggregator=aggregator)

    @staticmethod
    def tnr_parity(protected_col, X, y, pred, comparator="diff", aggregator="max", model=None):
        return FairnessMetrics.group_metric(FairnessMetrics.tnr, protected_col, X, y, pred, comparator=comparator,
                                            aggregator=aggregator)

    @staticmethod
    def class_change(x, options):
        options = options.copy()
        options.remove(x)
        return np.random.choice(options)

    @staticmethod
    def counter_factual_invariance(protected_col, X, y, pred, aggregator="min", model=None):
        assert model is not None
        aggregator = FairnessMetrics.get_aggregator(aggregator)
        n_classes_approx = y.nunique()
        options = X[protected_col].unique().tolist()
        counter_factual_equalities = []
        for n in range(n_classes_approx):
            cf = X.copy()
            cf[protected_col] = cf[protected_col].apply(lambda x: FairnessMetrics.class_change(x, options))
            cf_pred = model.predict(cf)
            eq = metrics.accuracy_score(pred, cf_pred)
            counter_factual_equalities.append(eq)
        return aggregator(counter_factual_equalities)

    @staticmethod
    def decoupled_regularization_loss(protected_col, X, y, pred, aggregator="sum", model=None):
        p_all = pred.mean()
        groups = X[protected_col].unique()
        group_losses = []
        for group in groups:
            idx = X[protected_col] == group
            p_k = pred[idx].mean()
            group_losses.append((p_k - p_all).abs())
        return aggregator(group_losses)

