import pandas as pd
import numpy as np
from models import *
from sklearn.metrics import accuracy_score, recall_score
from eval import *
from data import get_dataset
from dj_utils.utils import safe_mkdir
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def train_all(datasets=["compas", "german", "adults", "hmda"], save_dir="results/exhaustive/", n_folds=5,
              lambds=[0.2, 0.5, 0.7], pf_ms=ModelSets.best_models, rs_ms=ModelSets.best_models,
              dc_ms=[RandomForestClassifier(), XGBClassifier(use_label_encoder=False, eval_metric='logloss')],
              proba_models=ModelSets.best_models_proba, simple_models=ModelSets.best_models):
    skip_simple = len(simple_models) == 0
    skip_pf = len(pf_ms) == 0
    skip_rs = len(rs_ms) == 0
    skip_dcs = len(dc_ms) == 0
    skip_probas = len(proba_models) == 0
    for dataset in datasets:
        dataset = get_dataset(dataset)
        models = []
        true_model_length = 0
        if not skip_simple:
            base_models = []
            for i, model in enumerate(simple_models):
                basic = SKLearnModel(model=model, name=f"{model.__class__.__name__}_{i}",
                                     protected_col=dataset.protected_col)
                base_models.append(basic)
            models.extend(base_models)
            true_model_length += len(base_models)
        if not skip_pf:
            pf = ParetoFront(protected_col=dataset.protected_col, models=pf_ms)
            models.append(pf)
            true_model_length += 1
        if not skip_rs:
            rs = RegularizedSelection(protected_col=dataset.protected_col, models=rs_ms, lambd=lambds)
            models.append(rs)
            true_model_length += len(lambds)
        if not skip_dcs:
            dc = DecoupledClassifier(protected_col=dataset.protected_col, name=f"Decoupled_Classifier", models=dc_ms,
                                     lambd=lambds)
            models.append(dc)
            true_model_length += len(lambds)
        if not skip_probas:
            other_models = []
            for i, model in enumerate(proba_models):
                rpt = RecallParityThreshold(protected_col=dataset.protected_col, model=model,
                                            name=f"RPT_{model.__class__.__name__}_{i}")
                dms = [DatasetMassage(protected_col=dataset.protected_col, model=model, lambd=lambd,
                                      name=f"Massage_{model.__class__.__name__}_{i}_{lambd}") for lambd in lambds]
                other_models.append(rpt)
                other_models.extend(dms)
            models.extend(other_models)
            true_model_length += len(other_models)
        evaluator = Evaluator(dataset)
        print(f"{'X'*15}")
        print(f"Starting Process For {dataset.name}")
        with tqdm(total=true_model_length) as model_bar:
            for model in models:
                model_name = model.name
                if model.name == "Regularized_Selection" or model.name == "Decoupled_Classifier":
                    lambds = model.lambds
                    for lambd in lambds:
                        model_name = f"{model.name}_{lambd}"
                        evaluator.track_metric(tacc, accuracy_score, model_name=model_name)
                        evaluator.track_metric(vacc, accuracy_score, model_name=model_name)
                        evaluator.track_metric(recall, FairnessMetrics.recall_parity, model_name=model_name)
                        evaluator.track_metric(positive_rate, FairnessMetrics.positive_parity, model_name=model_name)
                        evaluator.track_metric(accuracy, FairnessMetrics.accuracy_parity, model_name=model_name)
                        evaluator.track_metric(tnr, FairnessMetrics.tnr_parity, model_name=model_name)
                        evaluator.track_metric(counterfactual, FairnessMetrics.counter_factual_invariance,
                                               model_name=model_name)
                else:
                    evaluator.track_metric(tacc, accuracy_score, model_name=model_name)
                    evaluator.track_metric(vacc, accuracy_score, model_name=model_name)
                    evaluator.track_metric(recall, FairnessMetrics.recall_parity, model_name=model_name)
                    evaluator.track_metric(positive_rate, FairnessMetrics.positive_parity, model_name=model_name)
                    evaluator.track_metric(accuracy, FairnessMetrics.accuracy_parity, model_name=model_name)
                    evaluator.track_metric(tnr, FairnessMetrics.tnr_parity, model_name=model_name)
                    evaluator.track_metric(counterfactual, FairnessMetrics.counter_factual_invariance,
                                           model_name=model_name)

                folds = dataset.get_cv_generators(folds=n_folds)
                for fold, gens in enumerate(folds):
                    train_gen, val_gen = gens
                    X_train, y_train = next(train_gen)
                    X_v, y_v = next(val_gen)

                    if model.name == "Regularized_Selection" or model.name == "Decoupled_Classifier":
                        model.fit(X_train, y_train)
                        for lambd in model.lambds:
                            pred_train = model.predict(X_train, lambd=lambd)
                            pred_v = model.predict(X_v, lambd=lambd)
                            model_name = f"{model.name}_{lambd}"

                            evaluator.update_sklearn_metric(y_train, pred_train, tacc, model_name=model_name, k=fold)
                            evaluator.update_sklearn_metric(y_v, pred_v, vacc, model_name=model_name, k=fold)
                            evaluator.update_fairness_metric(X_v, y_v, pred_v, name=recall, model_name=model_name, k=fold)
                            evaluator.update_fairness_metric(X_v, y_v, pred_v, name=positive_rate, model_name=model_name,
                                                             k=fold)
                            evaluator.update_fairness_metric(X_v, y_v, pred_v, name=accuracy, model_name=model_name, k=fold)
                            evaluator.update_fairness_metric(X_v, y_v, pred_v, name=tnr, model_name=model_name, k=fold)
                            evaluator.update_fairness_metric(X_v, y_v, pred_v, name=counterfactual, model_name=model_name,
                                                             k=fold, model=model)
                    else:
                        model.fit(X_train, y_train)
                        pred_train = model.predict(X_train)
                        pred_v = model.predict(X_v)

                        evaluator.update_sklearn_metric(y_train, pred_train, tacc, model_name=model_name, k=fold)
                        evaluator.update_sklearn_metric(y_v, pred_v, vacc, model_name=model_name, k=fold)
                        evaluator.update_fairness_metric(X_v, y_v, pred_v, name=recall, model_name=model_name, k=fold)
                        evaluator.update_fairness_metric(X_v, y_v, pred_v, name=positive_rate, model_name=model_name,
                                                         k=fold)
                        evaluator.update_fairness_metric(X_v, y_v, pred_v, name=accuracy, model_name=model_name, k=fold)
                        evaluator.update_fairness_metric(X_v, y_v, pred_v, name=tnr, model_name=model_name, k=fold)
                        evaluator.update_fairness_metric(X_v, y_v, pred_v, name=counterfactual, model_name=model_name,
                                                         k=fold, model=model)
                if model.name == "Regularized_Selection" or model.name == "Decoupled_Classifier":
                    model_bar.update(len(lambds))
                else:
                    model_bar.update()
                del folds
        evaluator.summarize_metrics(join=True)
        safe_mkdir(f"{save_dir}/{dataset.name}")
        evaluator.save(f"{save_dir}/")


if __name__ == "__main__":
    b = [RandomForestClassifier(),
          XGBClassifier(use_label_encoder=False, eval_metric='logloss')]
    train_all(datasets=["compas"], save_dir="results/tmp", pf_ms=b, rs_ms=b, dc_ms=b, proba_models=b, simple_models=b)



