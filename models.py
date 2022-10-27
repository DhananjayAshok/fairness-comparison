from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from eval import FairnessMetrics
from itertools import product
import numpy as np
import pandas as pd


class ModelSets:
    all_simple = [RidgeClassifier(), GaussianNB(), SVC(), MLPClassifier(),
                  RandomForestClassifier(),  XGBClassifier(use_label_encoder=False, eval_metric='logloss')]
    ridge_variants = [RidgeClassifier(alpha=al) for al in [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]]
    _svc_kernels = ["poly", "rbf", "sigmoid"]
    _svc_cs = [0.25, 0.5, 1]
    svc_variants = [SVC(C=c, kernel="rbf") for c in _svc_cs] + [SVC(C=c, kernel="sigmoid") for c in _svc_cs] + \
                   [SVC(C=c, kernel="poly", degree=3) for c in _svc_cs[::2]] + \
                   [SVC(C=c, kernel="poly", degree=4) for c in _svc_cs[::2]] + \
                   [SVC(C=c, kernel="poly", degree=5) for c in _svc_cs[::2]]
    random_forest_variants = [RandomForestClassifier(n_estimators=n) for n in [10, 50, 100, 200, 500]]
    xgboost_variants = [XGBClassifier(n_estimators=n) for n in [10, 50, 100, 200, 500]]
    all_variants = ridge_variants + svc_variants + random_forest_variants + xgboost_variants
    hidden_layer_sizes = [(100,), (100, 50), (1000, 100), (500,)]
    best_models = ridge_variants + [RandomForestClassifier(n_estimators=n) for n in [50, 100, 200]] + \
                  [XGBClassifier(n_estimators=n, use_label_encoder=False, eval_metric='logloss') for n in [50, 100, 200]] + [SVC(C=c, kernel="rbf") for c in _svc_cs] +\
                  [SVC(C=c, kernel="poly") for c in _svc_cs] + [GaussianNB()] + \
                  [MLPClassifier(hidden_layer_sizes=h) for h in hidden_layer_sizes]
    best_models_proba = [RandomForestClassifier(n_estimators=n) for n in [50, 100, 200]] + \
                  [XGBClassifier(n_estimators=n, use_label_encoder=False, eval_metric='logloss') for n in [50, 100, 200]] \
                        + [GaussianNB()] + [MLPClassifier(hidden_layer_sizes=h) for h in hidden_layer_sizes]
    best_trees = [RandomForestClassifier(n_estimators=n) for n in [50, 100, 200]] + \
                  [XGBClassifier(n_estimators=n, use_label_encoder=False, eval_metric='logloss') for n in [50, 100, 200]]


class AdjustmentModel:
    def __init__(self, name, protected_col):
        self.name = name
        self.protected_col = protected_col

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class SKLearnModel(AdjustmentModel):
    def __init__(self, model, name=None, protected_col=None):
        self.model = model
        if name is None:
            name = model.__class__.__name__
        AdjustmentModel.__init__(self, name=name, protected_col=protected_col)

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class ParetoFront(AdjustmentModel):
    def __init__(self, protected_col, name="Pareto", models=ModelSets.all_simple, metric=FairnessMetrics.recall_parity, metric_one_optimal=True,
                 metric_higher_is_better=False, train_size=0.8):
        AdjustmentModel.__init__(self, name, protected_col)
        self.models = models
        self.best_models = None
        self.train_size = train_size
        assert not metric_higher_is_better and metric_one_optimal
        # We will assume higher is better for self.metric_fn
        if metric_one_optimal:
            self.metric_fn = lambda X, y, pred: -1 * abs(metric(protected_col, X, y, pred)-1)
        else:
            if metric_higher_is_better:
                self.metric_fn = lambda X, y, pred: metric(protected_col, X, y)
            else:
                self.metric_fn = lambda X, y, pred: -1 * metric(protected_col, X, y)

    def fit(self, X, y):
        n_samples = int(self.train_size*len(y))
        X_train, y_train = X[:n_samples], y[:n_samples]
        X_v, y_v = X[n_samples:], y[n_samples:]
        fitted_models = []
        for model in self.models:
            fitted_model = model.fit(X_train, y_train)
            fitted_models.append(fitted_model)
        to_remove = []
        model_scores = []
        for fitted_model in fitted_models:
            pred = fitted_model.predict(X_v)
            acc = FairnessMetrics.accuracy(X_v, y_v, pred)
            metric = self.metric_fn(X_v, y_v, pred)
            model_scores.append((acc, metric))
        #  print_l = [f"{fitted_models[i].__class__.__name__}: acc {model_scores[i][0]} metric {model_scores[i][1]}"
        #             for i in range(len(models))]
        #  for l in print_l:
        #      print(l)
        for i, model_score in enumerate(model_scores):
            acc_i, metric_i = model_score
            for j, model_score_j in enumerate(model_scores):
                acc_j, metric_j = model_score_j
                if i == j or (fitted_models[i] in to_remove and fitted_models[j] in to_remove):
                    continue
                if fitted_models[i] not in to_remove:
                    if acc_i < acc_j and metric_i < metric_j:
                        to_remove.append(fitted_models[i])
                if fitted_models[j] not in to_remove:
                    if acc_i > acc_j and metric_i > metric_j:
                        to_remove.append(fitted_models[j])
        for rem in to_remove:
            if rem in fitted_models:
                fitted_models.remove(rem)
        self.best_models = fitted_models
        for model in self.best_models:
            model.fit(X, y)

    def predict(self, X):
        preds = np.zeros(len(X))
        for i, model in enumerate(self.best_models):
            preds += model.predict(X)
        preds = preds / len(self.best_models)
        return (preds >= 0.5).astype(int)


class RegularizedSelection(AdjustmentModel):
    def __init__(self, protected_col, name=f"Regularized_Selection", models=ModelSets.all_simple, lambd=0.8,
                 metric=FairnessMetrics.recall_parity, metric_one_optimal=True, train_size=0.8,
                 metric_higher_is_better=False):
        """

        :param protected_col:
        :param name:
        :param models:
        :param lambd: Can also be a list, if its a list then we will do a procedure for each lambda and save the res.
                    : unless specified explitly predictions are from the first lambda in the list.
        :param metric:
        :param metric_one_optimal:
        :param metric_higher_is_better:
        """
        AdjustmentModel.__init__(self, name, protected_col)
        self.models = models
        self.lambds = lambd
        self.train_size = train_size
        if type(lambd) != list:
            self.lambds = [lambd]
        else:
            self.lambds = lambd
        self.model_dict = {}
        assert not metric_higher_is_better and metric_one_optimal
        # We will assume higher is better for self.metric_fn
        if metric_one_optimal:
            self.metric_fn = lambda X, y, pred: -abs(metric(protected_col, X, y, pred) - 1)
        else:
            if metric_higher_is_better:
                self.metric_fn = lambda X, y, pred: metric(protected_col, X, y)
            else:
                self.metric_fn = lambda X, y, pred: -metric(protected_col, X, y)

    def loss_fn(self, X, y, pred, lambd):
        return -(lambd*(FairnessMetrics.accuracy(X, y, pred) + (1-lambd) * (self.metric_fn(X, y, pred))))

    def fit(self, X, y):
        n_samples = int(self.train_size*len(y))
        X_train, y_train = X[:n_samples], y[:n_samples]
        X_v, y_v = X[n_samples:], y[n_samples:]
        fitted_models = []
        model_preds = []
        for model in self.models:
            fitted_models.append(model.fit(X_train, y_train))
            model_preds.append(model.predict(X_v))

        for lambd in self.lambds:
            best_loss = np.inf
            best_model = None
            for i, fitted_model in enumerate(fitted_models):
                pred = model_preds[i]
                loss = self.loss_fn(X_v, y_v, pred, lambd)
                if loss < best_loss:
                    best_model = fitted_model
                    best_loss = loss
            self.model_dict[lambd] = best_model
            self.model_dict[lambd].fit(X, y)

    def get_model(self, lambd=None):
        if lambd is None or lambd not in self.model_dict:
            lambd = list(self.model_dict.keys())[0]
        model = self.model_dict[lambd]
        return model

    def predict(self, X, lambd=None):
        model = self.get_model(lambd)
        return model.predict(X)


class DecoupledClassifier(AdjustmentModel):
    def __init__(self, protected_col, name="Decoupled", models=ModelSets.all_simple, lambd=0.2,
                 reg=FairnessMetrics.decoupled_regularization_loss, reg_higher_worse=True):
        AdjustmentModel.__init__(self, name, protected_col)
        self.lambds = lambd
        if type(lambd) != list:
            self.lambds = [lambd]
        else:
            self.lambds = lambd
        self.models = models
        self.group_classifiers = {}
        for lamb in self.lambds:
            self.group_classifiers[lamb] = {}
        self.reg = reg
        self.reg_high_worse = reg_higher_worse

    def fit(self, X, y):
        groups = X[self.protected_col].unique()
        # We are assuming this will always be a representative sample
        model_set = {}
        for group in groups:
            idx = X[self.protected_col] == group
            group_X = X[idx]
            group_y = y[idx]
            model_set[group] = []
            for model in self.models:
                model_set[group].append(model.fit(group_X, group_y))
        model_lists = [model_set[group] for group in groups]
        mult = 1
        if not self.reg_high_worse:
            mult = -1
        for lambd in self.lambds:
            best_loss = np.inf
            best_group_models = None
            for group_models in product(*model_lists):
                pred = np.zeros_like(y)
                for group in groups:
                    idx = X[self.protected_col] == group
                    group_X = X[idx]
                    group_pred = group_models[group].predict(group_X)
                    pred[idx] = group_pred
                loss = lambd * FairnessMetrics.accuracy(X, y, pred) + (1-lambd) * mult * \
                       self.reg(self.protected_col, X, y, pred)
                if loss < best_loss:
                    best_loss = loss
                    best_group_models = group_models
            for group in groups:
                self.group_classifiers[lambd][group] = best_group_models[group]

    def get_group_classifier(self, lambd=None):
        if lambd is None or lambd not in self.group_classifiers:
            lambd = list(self.group_classifiers.keys())[0]
        group_clf = self.group_classifiers[lambd]
        return group_clf

    def predict(self, X, lambd=None):
        group_clf = self.get_group_classifier(lambd)
        pred = np.zeros(len(X))
        for group in group_clf:
            idx = X[self.protected_col] == group
            group_X = X[idx]
            if len(idx) == 0:
                continue
            model = group_clf[group]
            pred[idx] = model.predict(group_X)
        return pred


class RecallParityThreshold(AdjustmentModel):
    def __init__(self, protected_col,  model, name="RPT", delta=0.05):
        AdjustmentModel.__init__(self, protected_col=protected_col, name=name)
        self.model = model # Model should be instantiated not class
        self.delta = delta
        self.threshold_diff = {}

    def fit(self, X, y):
        self.model = self.model.fit(X, y)
        preds = self.model.predict(X)
        groups = X[self.protected_col].unique()
        assert len(groups) == 2
        g0, g1 = groups
        idx_0 = X[self.protected_col] == g0
        idx_1 = X[self.protected_col] == g1
        X_0, y_0, pred_0 = X[idx_0], y[idx_0], preds[idx_0]
        X_1, y_1, pred_1= X[idx_1], y[idx_1], preds[idx_1]
        r0 = FairnessMetrics.recall(X_0, y_0, pred_0)
        r1 = FairnessMetrics.recall(X_1, y_1, pred_1)
        if r1 > r0:
            g1, g0 = groups
            idx_0 = X[self.protected_col] == g0
            idx_1 = X[self.protected_col] == g1
            X_0, y_0 = X[idx_0], y[idx_0]
            X_1, y_1 = X[idx_1], y[idx_1]
            # Essentially r0 >= r1
        # print(f"Before Adjustment: {r0} vs {r1} were recalls")
        pred_0_proba = self.model.predict_proba(X_0)
        pred_1_proba = self.model.predict_proba(X_1)
        n = (0.5-2*self.delta)//self.delta
        for i in range(int(n)):
            p0 = 0.5 + self.delta * i
            p1 = 0.5 - self.delta * i
            pred_0 = (pred_0_proba[:, 1] >= p0).astype(int)
            pred_1 = (pred_1_proba[:, 1] >= p1).astype(int)
            self.threshold_diff[g0] = self.delta * i
            self.threshold_diff[g1] = -self.delta * i
            r0 = FairnessMetrics.recall(X_0, y_0, pred_0)
            r1 = FairnessMetrics.recall(X_1, y_1, pred_1)
            # print(f"At diff = {self.delta * i} recalls are {r0} vs {r1}")
            if r1 >= r0:
                break

    def predict(self, X):
        pred = np.zeros(len(X))
        for group in self.threshold_diff:
            idx = X[self.protected_col] == group
            pred[idx] = (self.model.predict_proba(X[idx])[:, 1] >= 0.5 + self.threshold_diff[group]).astype(int)
        return pred


class DatasetMassage(AdjustmentModel):
    def __init__(self, protected_col,  model, name="Massage", lambd=0.5):
        AdjustmentModel.__init__(self, protected_col=protected_col, name=name)
        self.model = model # Model should be instantiated not class
        self.lambd = lambd

    def fit(self, X, y):
        new_index = range(0, len(y))
        X = X.copy()
        y = y.copy()
        X.index = new_index
        y.index = new_index
        self.model = self.model.fit(X, y)
        preds = self.model.predict(X)
        groups = X[self.protected_col].unique()
        assert len(groups) == 2
        g0, g1 = groups
        idx_0 = X[self.protected_col] == g0
        idx_1 = X[self.protected_col] == g1
        X_0, y_0, pred_0 = X[idx_0], y[idx_0], preds[idx_0]
        X_1, y_1, pred_1= X[idx_1], y[idx_1], preds[idx_1]
        p0 = FairnessMetrics.positive_probability(X_0, y_0, pred_0)
        p1 = FairnessMetrics.positive_probability(X_1, y_1, pred_1)
        if p1 > p0:
            g1, g0 = groups
            idx_0 = X[self.protected_col] == g0
            idx_1 = X[self.protected_col] == g1
            X_0, y_0 = X[idx_0], y[idx_0]
            X_1, y_1 = X[idx_1], y[idx_1]
            # Essentially p0 >= p1
        pred_0_proba = self.model.predict_proba(X_0)[:, 1]
        pred_1_proba = self.model.predict_proba(X_1)[:, 1]
        discrimination = p1 - p0
        order_0 = np.argsort(pred_0_proba)
        order_1 = np.argsort(pred_1_proba)
        # Smallest first so we need to demote top D of 0 and promote bottom D of 1
        d = int(min(self.lambd * discrimination * len(X), min(len(idx_0), len(idx_1)) * 0.2))
        new_y = y.copy()
        #print(f"Got here with d={d}")
        if d > 0:
            new_y[order_0[0:d]] = 0
            new_y[order_1[-d:]] = 1
        self.model = self.model.fit(X, new_y)
        return

    def predict(self, X):
        return self.model.predict(X)


if __name__ == "__main__":
    from data import get_dataset
    from eval import Evaluator


