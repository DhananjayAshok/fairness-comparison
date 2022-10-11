import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from eval import Evaluator
from data import get_dataset
from dj_utils.utils import pbar
from tqdm import tqdm


dataset = get_dataset("adults")
models = [RidgeClassifier, SVC, RandomForestClassifier]
evaluator = Evaluator(dataset)
tacc = "Training Accuracy"
vacc = "Validation Accuracy"

for model_class in pbar(models, color="red", desc="Models", unit="model"):
    model_name = model_class.__name__
    evaluator.track_sklearn_metric(tacc, accuracy_score, model_name=model_name)
    evaluator.track_sklearn_metric(vacc, accuracy_score, model_name=model_name)

    folds = dataset.get_cv_generators(folds=5)
    for fold, gens in pbar(enumerate(folds), color="blue", desc="CV-Folds", unit="fold"):
        train_gen, val_gen = gens
        model = model_class()
        X_train, y_train = next(train_gen)
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        X_v, y_v = next(val_gen)
        pred_v = model.predict(X_v)
        evaluator.update_sklearn_metric(y_train, pred_train, tacc, model_name=model_name, k=fold)
        evaluator.update_sklearn_metric(y_train, pred_train, vacc, model_name=model_name, k=fold)

evaluator.summarize_sklearn_metric(tacc)
evaluator.summarize_sklearn_metric(vacc)
evaluator.plot_sklearn_metric_final(tacc)
evaluator.plot_sklearn_metric_final(vacc)



