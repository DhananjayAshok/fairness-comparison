import pandas as pd
import numpy as np
from models import *
from sklearn.metrics import accuracy_score, recall_score
from eval import Evaluator, FairnessMetrics
from data import get_dataset
from dj_utils.utils import safe_mkdir
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


datasets = ["hmda"]
for dataset in datasets:
    dataset = get_dataset(dataset)
    lamds = [0.2, 0.5, 0.7]
    pf = ParetoFront(protected_col=dataset.protected_col, models=ModelSets.all_simple)
    rss = [RegularizedSelection(protected_col=dataset.protected_col, name=f"Regularized_Selection_{lambd}", models=ModelSets.all_simple, lambd=lambd) for lambd in lamds]
    dcs = [DecoupledClassifier(protected_col=dataset.protected_col, name=f"Decoupled_Classifier_{lambd}", models=[RandomForestClassifier(), XGBClassifier(use_label_encoder=False, eval_metric='logloss')], lambd=lambd) for lambd in lamds]
    other_models = []
    for i, model in enumerate(ModelSets.best_models_proba):
        basic = SKLearnModel(model=model, name=f"{model.__class__.__name__}_{i}", protected_col=dataset.protected_col)
        rpt = RecallParityThreshold(protected_col=dataset.protected_col, model=model, name=f"RPT_{model.__class__.__name__}_{i}")
        dms = [DatasetMassage(protected_col=dataset.protected_col, model=model, lambd=lambd, name=f"Massage_{model.__class__.__name__}_{i}") for lambd in lamds]
        other_models.append(basic)
        other_models.append(rpt)
        other_models.extend(dms)
    models = [pf] + rss + dcs + other_models
    evaluator = Evaluator(dataset)
    tacc = "Training Accuracy"
    vacc = "Validation Accuracy"
    recall = "Validation Recall Parity"
    positive_rate = "Validation Positive Parity"
    accuracy = "Validation Accuracy Parity"
    tnr = "Validation TNR Parity"
    counterfactual = "Validation Counterfactual Invariance"
    n_folds = 5
    print(f"{'X'*15}")
    print(f"Starting Process For {dataset.name}")
    with tqdm(total=len(models)) as model_bar:
        for model in models:
            model_name = model.name
            evaluator.track_metric(tacc, accuracy_score, model_name=model_name)
            evaluator.track_metric(vacc, accuracy_score, model_name=model_name)
            evaluator.track_metric(recall, FairnessMetrics.recall_parity, model_name=model_name)
            evaluator.track_metric(positive_rate, FairnessMetrics.positive_parity, model_name=model_name)
            evaluator.track_metric(accuracy, FairnessMetrics.accuracy_parity, model_name=model_name)
            evaluator.track_metric(tnr, FairnessMetrics.tnr_parity, model_name=model_name)
            evaluator.track_metric(counterfactual, FairnessMetrics.counter_factual_invariance, model_name=model_name)

            folds = dataset.get_cv_generators(folds=n_folds)
            for fold, gens in enumerate(folds):
                train_gen, val_gen = gens
                X_train, y_train = next(train_gen)
                model.fit(X_train, y_train)
                pred_train = model.predict(X_train)
                X_v, y_v = next(val_gen)
                pred_v = model.predict(X_v)
                evaluator.update_sklearn_metric(y_train, pred_train, tacc, model_name=model_name, k=fold)
                evaluator.update_sklearn_metric(y_v, pred_v, vacc, model_name=model_name, k=fold)
                evaluator.update_fairness_metric(X_v, y_v, pred_v, name=recall, model_name=model_name, k=fold)
                evaluator.update_fairness_metric(X_v, y_v, pred_v, name=positive_rate, model_name=model_name, k=fold)
                evaluator.update_fairness_metric(X_v, y_v, pred_v, name=accuracy, model_name=model_name, k=fold)
                evaluator.update_fairness_metric(X_v, y_v, pred_v, name=tnr, model_name=model_name, k=fold)
                evaluator.update_fairness_metric(X_v, y_v, pred_v, name=counterfactual, model_name=model_name, k=fold,
                                                 model=model)

            del folds
            model_bar.update()

    evaluator.summarize_metrics()
    safe_mkdir(f"results/{dataset.name}")
    evaluator.save(f"results/{dataset.name}")
    #evaluator.plot_metric_final(tacc)
    #evaluator.plot_metric_final(vacc)
    #evaluator.plot_metric_final(recall)
    #evaluator.plot_metric_final(counterfactual)
    #evaluator.plot_metrics_final(vacc, recall)
    #evaluator.plot_metrics_final(vacc, counterfactual)


