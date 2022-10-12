import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score
from eval import Evaluator, FairnessMetrics
from data import get_dataset
from dj_utils.utils import pbar
from tqdm import tqdm


dataset = get_dataset("adults")
models = [RidgeClassifier, RandomForestClassifier , SVC,  NearestCentroid, MLPClassifier, GaussianNB, XGBClassifier]
evaluator = Evaluator(dataset)
tacc = "Training Accuracy"
vacc = "Validation Accuracy"
recall = "Validation Recall Parity"

for model_class in tqdm(models):
    model_name = model_class.__name__
    evaluator.track_metric(tacc, accuracy_score, model_name=model_name)
    evaluator.track_metric(vacc, accuracy_score, model_name=model_name)
    evaluator.track_metric(recall, FairnessMetrics.recall_parity, model_name=model_name)

    folds = dataset.get_cv_generators(folds=5)
    for fold, gens in tqdm(enumerate(folds)):
        train_gen, val_gen = gens
        model = model_class()
        X_train, y_train = next(train_gen)
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        X_v, y_v = next(val_gen)
        pred_v = model.predict(X_v)
        evaluator.update_sklearn_metric(y_train, pred_train, tacc, model_name=model_name, k=fold)
        evaluator.update_sklearn_metric(y_v, pred_v, vacc, model_name=model_name, k=fold)
        evaluator.update_fairness_metric(X_v, y_v, pred_v, name=recall, model_name=model_name, k=fold)
    del folds

evaluator.summarize_metrics()
evaluator.plot_metric_final(tacc)
evaluator.plot_metric_final(vacc)
evaluator.plot_metric_final(recall)
evaluator.plot_metrics_final(vacc, recall)
