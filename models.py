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


class ParetoFront(AdjustmentModel):
    def __init__(self, protected_col, name="Pareto", model_classes=[RidgeClassifier, GaussianNB,
                                     XGBClassifier], metric=FairnessMetrics.recall_parity, metric_one_optimal=True,
                 metric_higher_is_better=False):
        AdjustmentModel.__init__(self, name, protected_col)
        self.model_classes = model_classes
        self.best_models = None
        assert not metric_higher_is_better and metric_one_optimal
        # We will assume higher is better for self.metric_fn
        if metric_one_optimal:
            self.metric_fn = lambda X, y, pred: -(metric(protected_col, X, y, pred)-1).abs()
        else:
            if metric_higher_is_better:
                self.metric_fn = lambda X, y, pred: metric(protected_col, X, y)
            else:
                self.metric_fn = lambda X, y, pred: -metric(protected_col, X, y)

    def fit(self, X, y):
        models = []
        for model_class in self.model_classes:
            model = model_class().fit(X, y)
            models.append(model)
        to_remove = []
        model_scores = []
        for model in models:
            pred = model.predict(X)
            acc = FairnessMetrics.accuracy(X, y, pred)
            metric = self.metric_fn(X, y, pred)
            model_scores.append((acc, metric))
        for i, model_score in enumerate(model_scores):
            acc_i, metric_i = model_score
            for j, model_score_j in enumerate(model_scores):
                acc_j, metric_j = model_score_j
                if i == j or (models[i] in to_remove and models[j] in to_remove):
                    continue
                if models[i] not in to_remove:
                    if acc_i < acc_j and metric_i < metric_j:
                        to_remove.append(models[i])
                if models[j] not in to_remove:
                    if acc_i > acc_j and metric_i > metric_j:
                        to_remove.append(models[j])
        for rem in to_remove:
            if rem in models:
                models.remove(rem)
        self.best_models = models

    def predict(self, X):
        preds = np.zeros((len(X), len(self.best_models)))
        for i, model in enumerate(self.best_models):
            preds[:, i] = model.predict(X)
        return (preds.sum(axis=1) > len(preds) / 2).astype(int)


class RegularizedSelection(AdjustmentModel):
    def __init__(self, protected_col, name=f"Regularized_Selection", model_classes=[RidgeClassifier, GaussianNB,
                                     XGBClassifier], lambd=0.8,
                 metric=FairnessMetrics.recall_parity, metric_one_optimal=True,
                 metric_higher_is_better=False):
        AdjustmentModel.__init__(self, name, protected_col)
        self.model_classes = model_classes
        self.lambd = lambd
        self.model = None
        assert not metric_higher_is_better and metric_one_optimal
        # We will assume higher is better for self.metric_fn
        if metric_one_optimal:
            self.metric_fn = lambda X, y, pred: -(metric(protected_col, X, y, pred) - 1).abs()
        else:
            if metric_higher_is_better:
                self.metric_fn = lambda X, y, pred: metric(protected_col, X, y)
            else:
                self.metric_fn = lambda X, y, pred: -metric(protected_col, X, y)

    def loss_fn(self, X, y, pred):
        return -(self.lambd*(FairnessMetrics.accuracy(X, y, pred) + (1-self.lambd) * (self.metric_fn(X, y, pred))))

    def fit(self, X, y):
        best_loss = np.inf
        best_model = None
        for model in self.model_classes:
            fitted_model = model.fit(X, y)
            pred = fitted_model(X)
            loss = self.loss_fn(X, y, pred)
            if loss < best_loss:
                best_model = fitted_model
                best_loss = loss
        self.model = best_model
        self.name = f"{self.name}_{self.model.__class__.__name__}"

    def predict(self, X):
        return self.model.predict(X)


class DecoupledClassifier(AdjustmentModel):
    def __init__(self, protected_col, name="Decoupled",  model_classes=[RidgeClassifier, GaussianNB,
                                     XGBClassifier], lambd=0.2, reg=FairnessMetrics.decoupled_regularization_loss,
                 reg_higher_worse=True):
        AdjustmentModel.__init__(self, name, protected_col)
        self.model_classes = model_classes
        self.lambd = lambd
        self.group_classifiers = {}
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
            for model_class in self.model_classes:
                model_set[group] = model_class().fit(group_X, group_y)
        model_lists = [model_set[group] for group in groups]
        best_loss = np.inf
        best_group_models = None
        mult = 1
        if not self.reg_high_worse:
            mult = -1
        for group_models in product(*model_lists):
            pred = np.zeros_like(y)
            for group in groups:
                idx = X[self.protected_col] == group
                group_X = X[idx]
                group_pred = group_models[group].predict(group_X)
                pred[idx] = group_pred
            loss = self.lambd * FairnessMetrics.accuracy(X, y, pred) + (1-self.lambd) * mult * \
                   self.reg(self.protected_col, X, y, pred)
            if loss < best_loss:
                best_loss = loss
                best_group_models = group_models
        for group in groups:
            self.group_classifiers[group] = best_group_models[group]

    def predict(self, X):
        pred = np.zeros(len(X))
        for group in self.group_classifiers:
            idx = X[self.protected_col] == group
            group_X = X[idx]
            if len(idx) == 0:
                continue
            model = self.group_classifiers[group]
            pred[idx] = model(group_X)
        return pred


class RecallParityThreshold(AdjustmentModel):
    def __init__(self, protected_col,  model, name="Threshold", delta=0.05):
        AdjustmentModel.__init__(protected_col=protected_col, name=name)
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
        pred_0_proba = self.model.predict_proba(X_0)
        pred_1_proba = self.model.predict_proba(X_1)
        n = (0.5-2*self.delta)//self.delta
        for i in range(n):
            p0 = 0.5 + self.delta * i
            p1 = 0.5 - self.delta * i
            pred_0 = (pred_0_proba >= p0).astype(int)
            pred_1 = (pred_1_proba >= p1).astype(int)
            self.threshold_diff[g0] = self.delta * i
            self.threshold_diff[g1] = -self.delta * i
            r0 = FairnessMetrics.recall(X_0, y_0, pred_0)
            r1 = FairnessMetrics.recall(X_1, y_1, pred_1)
            if r1 >= r0:
                break

    def predict(self, X):
        pred = np.zeros(len(X))
        for group in self.threshold_diff:
            idx = X[self.protected_col] == group
            pred[idx] = (self.model.predict_proba(X[idx]) >= 0.5 + self.threshold_diff[group]).astype(int)
        return pred


class DatasetMassage(AdjustmentModel):
    def __init__(self, protected_col,  model, name="Massage", lambd=0.5):
        AdjustmentModel.__init__(protected_col=protected_col, name=name)
        self.model = model # Model should be instantiated not class
        self.lambd = lambd

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
        p0 = FairnessMetrics.positive_probability(X_0, y_0, pred_0)
        p1 = FairnessMetrics.positive_probability(X_1, y_1, pred_1)
        if p1 > p0:
            g1, g0 = groups
            idx_0 = X[self.protected_col] == g0
            idx_1 = X[self.protected_col] == g1
            X_0, y_0 = X[idx_0], y[idx_0]
            X_1, y_1 = X[idx_1], y[idx_1]
            # Essentially p0 >= p1
        pred_0_proba = self.model.predict_proba(X_0)
        pred_1_proba = self.model.predict_proba(X_1)
        discrimination = p1 - p0
        order_0 = np.argsort(pred_0_proba)
        order_1 = np.argsort(pred_1_proba)
        # Smallest first so we need to demote top D of 0 and promote bottom D of 1
        d = int(self.lambd * discrimination * len(X))
        new_y = y.copy()
        if d > 0:
            new_y[order_0[0:d]] = 0
            new_y[order_1[-d:]] = 1
        self.model = self.model.fit(X, new_y)
        return

    def predict(self, X):
        return self.model.predict(X)