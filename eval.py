import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

tacc = "Training Accuracy"
vacc = "Validation Accuracy"
recall = "Validation Recall Parity"
positive_rate = "Validation Positive Parity"
accuracy = "Validation Accuracy Parity"
tnr = "Validation TNR Parity"
counterfactual = "Validation Counterfactual Invariance"


class Evaluator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.fold_results = {}
        self.df = None

    def track_metric(self, name, metric_fn, model_name=None):
        """
        Meant to store
        :param name:
        :param model_name: should not be 'metric'
        :param metric_fn: takes in y_true, y_pre
        :return:
        """
        if name in self.fold_results:
            assert model_name is not None
            self.fold_results[name][model_name] = []
            assert metric_fn == self.fold_results[name]['metric']
        else:
            if model_name is None:
                self.fold_results[name] = {"metric": metric_fn}
            else:
                self.fold_results[name] = {"metric": metric_fn, model_name: []}

    def update_sklearn_metric(self, y_true, y_pred, name, model_name):
        """

        :param y_true:
        :param y_pred:
        :param name:
        :param model_name:
        :return:
        """
        assert name in self.fold_results
        val = self.fold_results[name]['metric'](y_true, y_pred)
        self.direct_update_metric(val, name, model_name)

    def update_fairness_metric(self, X, y, pred, name, model_name, model=None):
        assert name in self.fold_results
        val = self.fold_results[name]['metric'](self.dataset.protected_col, X, y, pred, model=model)
        self.direct_update_metric(val, name, model_name)

    def direct_update_metric(self, val, name, model_name):
        assert name in self.fold_results
        assert model_name in self.fold_results[name]
        self.fold_results[name][model_name].append(val)
        return

    def summarize_metric(self, name):
        assert name in self.fold_results
        f_res = self.fold_results[name]
        f_res.pop('metric')
        data = []
        columns = ["Fold", name, "Model"]
        for model_name in f_res.keys():
            folds = f_res[model_name]
            for fold in range(len(folds)):
                data.append([fold, folds[fold], model_name])
        df = pd.DataFrame(data=data, columns=columns)
        self.fold_results[name] = df

    def summarize_metrics(self, join=True):
        for name in self.fold_results:
            self.summarize_metric(name)
        if join:
            self.join_trajectories()

    def join_trajectories(self):
        if self.df is None:
            metrics = list(self.fold_results.keys())
            m = metrics[0]
            base_df = self.fold_results[m]
            for metric in metrics[1:]:
                base_df[metric] = self.fold_results[metric][metric]
                self.fold_results[metric] = None
            self.df = base_df

    def save(self, path):
        if self.df is None:
            for name in self.fold_results:
                self.fold_results[name].to_csv(os.path.join(path, f"{name}.csv"), index=False)
        else:
            self.df.to_csv(os.path.join(path, f"{self.dataset.name}.csv"), index=False)

    def load(self, path):
        self.fold_results = {}
        files = os.listdir(path)
        if f"{self.dataset.name}.csv" in files:
            self.df = pd.read_csv(os.path.join(path, f"{self.dataset.name}.csv"))
            return
        else:
            names = os.listdir(os.path.join(path, self.dataset.name))
            for name in names:
                # .csv excluded so -4
                df = pd.read_csv(os.path.join(path, self.dataset.name, name))
                if 'Unnamed: 0' in df:
                    df.drop('Unnamed: 0', axis=1, inplace=True)
                self.fold_results[name[:-4]] = df

    def select_model_subset_df(self, name, model_subset):
        if self.df is None:
            assert name in self.fold_results
            df = self.fold_results[name]
            if model_subset is None:
                model_subset = df["Model"].unique().tolist()
            else:
                model_subset = list(df["Model"].unique().toset().intersect(set(model_subset)))
            assert len(model_subset) > 0
            df = df[df["Model"].isin(model_subset)]
        else:
            df = self.df[self.df["Model"].isin(model_subset)]
            df = df[["Fold", "Epoch", name, "Model"]]
        return df

    def select_model_subset_df_multi(self, names, model_subset):
        """
        Assumes all have same epochs, folds, models etc.
        :param names:
        :param model_subset:
        :return:
        """
        if self.df is None:
            assert all([name in self.fold_results for name in names])
            df = self.fold_results[names[0]]
            if model_subset is None:
                model_subset = df["Model"].unique().tolist()
            else:
                model_subset = list(df["Model"].unique().toset().intersect(set(model_subset)))
            assert len(model_subset) > 0
            for name in names[1:]:
                df[name] = self.fold_results[name][name]
            df = df[df["Model"].isin(model_subset)]
        else:
            df = self.df[self.df["Model"].isin(model_subset)]
            df = df[["Fold", "Epoch", "Model"] + names]
        return df

    def plot_metric(self, name, model_subset=None, use_base_model=False):
        df = None
        if self.df is None:
            assert name in self.fold_results and type(self.fold_results[name]) == pd.DataFrame
            df = self.select_model_subset_df(name, model_subset)
        else:
            df = self.df
        df = df.copy()
        if use_base_model:
            df["Model"] = df["Model"].apply(Evaluator.get_base_model)
        sns.boxplot(data=df, x="Model", y=name)
        plt.show()
        return

    def plot_metrics_final(self, name_0, name_1, model_subset=None, use_base_model=False):
        df = None
        if self.df is None:
            assert [name in self.fold_results and type(self.fold_results[name]) == pd.DataFrame for name in
                    [name_0, name_1]]
            df = self.select_model_subset_df_multi([name_0, name_1], model_subset=model_subset)
        else:
            df = self.df
        df = df.copy()
        if use_base_model:
            df["Model"] = df["Model"].apply(Evaluator.get_base_model)
        sns.scatterplot(data=df, x=name_0, y=name_1, hue="Model", style="Model")
        plt.show()
        return

    def get_top_model_indices(self, metric, k=10, ascending=True, one_optimal=False):
        """

        :param self:
        :param metric:
        :param k:
        :param ascending:
        :param one_optimal: if One Optimal is True then disregard descending flag
        :return: index that will sort by this
        """
        if self.df is None:
            df = self.fold_results[metric].copy()
        else:
            df = self.df[metric]
        entries = None
        if one_optimal:
            df[f"abs({metric}-1)"] = abs(df[metric] - 1)
            entries = df.sort_values(f"abs({metric}-1)", axis=1)
        else:
            entries = df.sort_values(metric, ascending=ascending, axis=1)
        return entries[:k]

    @staticmethod
    def get_base_model(string):
        from models import ModelSets
        phrase_returns = ["Pareto", "Decoupled"]
        for phrase in phrase_returns:
            if phrase in string:
                return phrase
        for model in ModelSets.all_simple:
            if model.__class__.__name__ in string:
                return model.__class__.__name__
        if "Regularized_Selection" in string:
            return "Regularized_Selection"
        else:
            return "?"

    @staticmethod
    def a_better_or_equal_b(a, b, opt):
        if opt == "a":
            return a <= b
        elif opt == "d":
            return a >= b
        elif opt == "o":
            return abs(a - 1) <= abs(b - 1)
        else:
            raise ValueError

    @staticmethod
    def a_worse_b(a, b, opt):
        return not Evaluator.a_better_or_equal_b(a, b, opt)

    def get_pareto_tiers(self, metrics, optimalities, use_base_model=False, df=None):
        """
        Returns a list of lists of models: models in list i are dominated by models in list i-1
        :param self:
        :param metrics: list of metric names strings
        :param optimalities: list of either 'a', 'd', or 'o'=> ascending (lower better), descending or one is optimal
        :return:
        Assumes a evaluator.join_trajectory has been run for this evaluator / the load was from a single csv
        Computes for mean metric value
        """
        assert self.df is not None
        dominated_models = []
        if df is None:
            df = self.df.copy()
            if use_base_model:
                df["Model"] = df["Model"].apply(Evaluator.get_base_model)
            df = df.groupby("Model").mean().drop("Fold", axis=1)
        models = df.index
        for model in models:
            for other_model in models:
                dominated = all([Evaluator.a_worse_b(df.loc[model, metrics[i]], df.loc[other_model, metrics[i]],
                                                     optimalities[i]) for i in range(len(metrics))])
                if dominated:
                    dominated_models.append(model)
                    break
        if len(dominated_models) == 0:
            return [list(models)]
        else:
            better_df = df.drop(dominated_models, axis=0)
            dominated_df = df.loc[dominated_models, :]
            better_model_tiers = self.get_pareto_tiers(metrics=metrics, optimalities=optimalities, df=better_df)
            worse_model_tiers = self.get_pareto_tiers(metrics=metrics, optimalities=optimalities, df=dominated_df)
            return better_model_tiers + worse_model_tiers

    def display_pareto_tiers(self, metrics, optimalities, use_base_model=False, use_base_model_after=False):
        p_tiers = self.get_pareto_tiers(metrics=metrics, optimalities=optimalities, use_base_model=use_base_model)
        df = self.df.copy()
        if use_base_model or use_base_model_after:
            df["Model"] = df["Model"].apply(Evaluator.get_base_model)
        df = df.groupby("Model").mean().drop("Fold", axis=1)[metrics]
        for i, tier in enumerate(p_tiers):
            if use_base_model_after and not use_base_model:
                tier = [Evaluator.get_base_model(model) for model in tier]
            print(f"Tier: {i+1}")
            print(df.loc[tier, :])


    @staticmethod
    def get_all_evaluators(datasets=["adults", "compas", "german", "hmda"], path="results/"):
        from data import get_dataset
        res = {}
        for dataset in datasets:
            dset = get_dataset(dataset)
            path = os.path.join(path)
            evaluator = Evaluator(dataset=dset)
            evaluator.load(path)
            res[dataset] = evaluator
        return res


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
        aggregator = FairnessMetrics.get_aggregator(aggregator)
        p_all = pred.mean()
        groups = X[protected_col].unique()
        group_losses = []
        for group in groups:
            idx = X[protected_col] == group
            p_k = pred[idx].mean()
            group_losses.append(abs(p_k - p_all))
        return aggregator(group_losses)


def quick_eval():
    evals = Evaluator.get_all_evaluators(datasets=["german"], path="results/trees")["german"]
    return evals


def tmp():
    metrics = [vacc, recall]
    optimalities = ["d", "o"]
    return metrics, optimalities


if __name__ == "__main__":
    evals = Evaluator.get_all_evaluators(path="results/trees/")
    for dset in evals:
        print(f"Working on {dset}")
        e = evals[dset]
        e.df["Fold"] = e.df["Epoch"]
        e.df = e.df.drop("Epoch", axis=1)
        e.save("results/trees")

