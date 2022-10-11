import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Evaluator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.trajectories = {}

    def track_sklearn_metric(self, name, metric_fn, model_name=None):
        """
        Meant to store
        :param name:
        :param model_name: should not be 'metric'
        :param metric_fn: dee get_sklearn_metric
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
        self.direct_update_sklearn_metric(val, name, model_name, k=k)

    def direct_update_sklearn_metric(self, val, name, model_name, k=None):
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

    def summarize_sklearn_metric(self, name):
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


    def plot_sklearn_metric_trajectory(self, name, model_subset=None):
        assert name in self.trajectories and type(self.trajectories[name]) == pd.DataFrame
        df = self.select_model_subset_df(name, model_subset)
        sns.lineplot(data=df, x="Epoch", y=name, hue="Model")
        plt.show()
        return

    def plot_sklearn_metric_final(self, name, model_subset=None):
        assert name in self.trajectories and type(self.trajectories[name]) == pd.DataFrame
        df = self.select_model_subset_df(name, model_subset)
        last_epoch = df["Epoch"].max()
        last_only = df[df["Epoch"] == last_epoch]
        sns.barplot(data=last_only, x="Model", y=name)
        plt.show()
        return

    def plot_sklearn_metrics_trajectory(self, names, model_subset=None):
        df = self.select_model_subset_df_multi(names, model_subset)
        pass

