from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
import joblib
import optuna
from optuna.samplers import TPESampler
from objective import fixed_ml_hyperparams
import os
from utils import set_seed, save_yaml
import logging
import sys

SEED=1
set_seed(SEED)

def cv_folds(X, y, groups, n_splits, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fold_files = [f"{save_dir}/fold_{i}.pkl" for i in range(n_splits)]
    if all(os.path.exists(f) for f in fold_files):
        print(f"[✓] Saved {n_splits}-fold data has been detected and will be loaded directly")
        folds = []
        for f in fold_files:
            fold_data = joblib.load(f)
            folds.append((fold_data["train_idx"], fold_data["valid_idx"]))
        return folds  
    print(f"[✳] Generate {n_splits} fold cross-validation and save...")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=1)
    folds = list(cv.split(X, y, groups))
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        fold_data = {
            "train_idx": train_idx,
            "valid_idx": test_idx,
        }
        joblib.dump(fold_data, f"{save_dir}/fold_{fold_idx}.pkl")
    print(f"Saved {n_splits}-fold splits to {save_dir}")
    return folds

def run_model_pipeline(
    X_train,
    y_train,
    X_test, 
    y_test,
    folds,
    model_type, # "dl" or "ml"
    model_name,  # e.g. e.g., "knn" for ML or "cnn" for DL
    objective_fn, # e.g., objective_ml_model or objective_dl_model
    test_fn, # e.g., test_ml_model or test_pytorch_model
    seed,
    search_space=None,
    n_trials=20,
    score_metric="auroc",
    save_path="./out/Prostate_cancer",
    device="cpu",
    ):

    set_seed(seed)
    
    os.makedirs(save_path, exist_ok=True)
    '''
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)  
    scaler_save_path = f"{save_path}/model/scaler/scaler.save"
    if not os.path.exists(scaler_save_path):
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
    '''     
    # Construct the Optuna search space and commence optimization
    optuna.logging.disable_default_handler()
    optuna_logger = optuna.logging.get_logger("optuna")
    optuna_logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    if optuna_logger.hasHandlers():
        optuna_logger.handlers.clear()
    optuna_logger.addHandler(console_handler)
    
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"search_for_{model_name}")

    def trial_objective(trial):
        kwargs = {
            "trial": trial,
            "model_name": model_name,
            "X": X_train,
            "y": y_train,
            "IsTrial": True,
            "folds": folds,
            "search_params": search_space,
            "score_metric": score_metric,
            "save_path": save_path,
            "device": device if model_type == "dl" else None
        }
        return objective_fn(**{k: v for k, v in kwargs.items() if v is not None})

    study.optimize(trial_objective, n_trials=n_trials)

    # Save best parameters
    final_params = study.best_params.copy()

    default_space = fixed_ml_hyperparams(model_name,seed) or {}

    if search_space is not None:
        for k, v in search_space.items():
            if not isinstance(v, dict): 
                final_params[k] = v

        for k, v in default_space.items():
            if not isinstance(v, dict) and k not in final_params:
                final_params[k] = v
    else:

        for k, v in default_space.items():
            if not isinstance(v, dict):
                final_params[k] = v  
                
    save_model_path = f"{save_path}/model/{model_name}"
    os.makedirs(save_model_path, exist_ok=True) 
    save_yaml(final_params, 
              f"{save_model_path}/best_search_params.yaml")

    # Train the final model using the entire training set
    final_kwargs = {
        "trial": None,
        "model_name": model_name,
        "X": X_train,
        "y": y_train,
        "IsTrial": False,
        "folds": folds,
        "search_params": final_params,
        "score_metric": score_metric,
        "save_path": save_path,
    }
    if model_type == "dl":
        final_kwargs["device"] = device
    final_score, model = objective_fn(**final_kwargs)

    # Test set evaluation
    test_kwargs = {
        "model": model,
        "model_name": model_name,
        "score_metric": score_metric,
        "X_test": X_test,
        "y_test": y_test,
        "save_path": save_path
    }
    if model_type == "dl":
        test_kwargs["device"] = device
    test_score = test_fn(**test_kwargs)
    

    return {
        "best_cv_score": study.best_value,
        "best_params": study.best_params,
        "final_train_score": final_score,
        "test_score": test_score
    }
 