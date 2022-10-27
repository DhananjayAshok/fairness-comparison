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


def train_all(datasets=["compas", "german", "adults", "hmda"], save_dir="results/exhaustive/", n_folds=5, lambds=[0.2, 0.5, 0.7],
          pf_ms=ModelSets.best_models, rs_ms=ModelSets.best_models, dc_ms=[RandomForestClassifier(),
          XGBClassifier(use_label_encoder=False, eval_metric='logloss')], proba_models=ModelSets.best_models_proba,
              simple_models=ModelSets.best_models):
    skip_simple = len(simple_models) == 0
    skip_pf = len(pf_ms) == 0
    skip_rs = len(rs_ms) == 0
    skip_dcs = len(dc_ms) == 0
    skip_probas = len(proba_models) == 0
    for dataset in datasets:
        dataset = get_dataset(dataset)
        models = []
        if not skip_simple:
            base_models = []
            for i, model in enumerate(simple_models):
                basic = SKLearnModel(model=model, name=f"{model.__class__.__name__}_{i}",
                                     protected_col=dataset.protected_col)
                base_models.append(basic)
            models.extend(base_models)
        if not skip_pf:
            pf = ParetoFront(protected_col=dataset.protected_col, models=[pf_ms])
            models.append(pf)
        if not skip_rs:
            rs = RegularizedSelection(protected_col=dataset.protected_col, models=rs_ms, lambd=lambds)
            models.append(rs)
        if not skip_dcs:
            dcs = [DecoupledClassifier(protected_col=dataset.protected_col, name=f"Decoupled_Classifier_{lambd}",
                                       models=dc_ms, lambd=lambd) for lambd in lambds]
            models.extend(dcs)
        if not skip_probas:
            proba_models = []
            for i, model in enumerate(proba_models):
                rpt = RecallParityThreshold(protected_col=dataset.protected_col, model=model,
                                            name=f"RPT_{model.__class__.__name__}_{i}")
                dms = [DatasetMassage(protected_col=dataset.protected_col, model=model, lambd=lambd,
                                      name=f"Massage_{model.__class__.__name__}_{i}") for lambd in lamds]
                proba_models.append(rpt)
                proba_models.extend(dms)
            models.extend(proba_models)
        evaluator = Evaluator(dataset)
        print(f"{'X'*15}")
        print(f"Starting Process For {dataset.name}")
        with tqdm(total=len(models)) as model_bar:
            for model in models:
                model_name = model.name
                if model.name == "Regularized_Selection":
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

                    if model.name == "Regularized_Selection":
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
                del folds
                model_bar.update()
        evaluator.summarize_metrics(join=True)
        safe_mkdir(f"{save_dir}/{dataset.name}")
        evaluator.save(f"{save_dir}/")




