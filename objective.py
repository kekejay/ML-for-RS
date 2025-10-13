import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
import numpy as np
import os
import pandas as pd
from utils import evaluate, defined_metric_score, save_ml_model
import torch
from torch.utils.data import DataLoader, TensorDataset
from trainer import train_pytorch_model
from models import CNN1D, LSTM1D, GRU1D, Transformer1D, Mamba1D


def customize_param(trial, search_space, param_name):
    param_info = search_space[param_name]
    t = param_info.get("type")
    args = param_info.get("args", ())

    if t == "categorical":
        return trial.suggest_categorical(param_name, args)
    elif t == "int":
        return trial.suggest_int(param_name, args[0], args[1])
    elif t == "float":
        log = param_info.get("log", False)
        return trial.suggest_float(param_name, args[0], args[1], log=log)
    else:
        return args

def fixed_ml_hyperparams(model_name,seed):
    if model_name == "xgb":
        return {
            "device": "cuda",
            "tree_method": "hist",
            "objective": "binary:logistic",
            "seed": seed
        }
    if model_name == "logreg":
        return {
            "random_state": seed,
            "solver": "liblinear"
        }    
    if model_name in ["lgbm", "rf", "gdbt"]:
        return {
            "random_state": seed,
        }
    if model_name == "pls_da":
        return {
            "clf__max_iter": 500,
        }    
                
def suggest_ml_hyperparams(trial, model_name):
    if model_name == "xgb":
        return {
            "device":"cuda",
            "tree_method":"hist",
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "objective": "binary:logistic",
            "seed": 1
        }
    elif model_name == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 1
        }
    elif model_name.startswith("svm"):
        param = {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
            "gamma": 'scale'
            }
        if param["kernel"] == "rbf":
            param["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto", 0.001, 0.01, 0.1, 1])
        return param
    elif model_name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "random_state": 1,
        }
    elif model_name == "gbdt":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "random_state": 1
        }
    elif model_name == "logreg":
        return {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "solver": "liblinear",
            "random_state": 1
        }
    elif model_name == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 15),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 2)
        }
    elif model_name == "pca_lda":
        solver = trial.suggest_categorical("lda__solver", ["svd", "lsqr"])
        params = {
            "pca__n_components": trial.suggest_int("pca__n_components", 2, 50),
            "lda__solver": solver,
        }
        if solver == "lsqr":
            params["lda__shrinkage"] = trial.suggest_categorical("lda__shrinkage", ["auto", None])
        params["random_state"] = 1    
        return params
    elif model_name == "pls_da":
        return {
            "pls__n_components": trial.suggest_int("pls__n_components", 2, 50),
            "clf__C": trial.suggest_float("clf__C", 1e-3, 10.0, log=True),
            #"clf__max_iter": 500
        }
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    
def suggest_dl_hyperparams(trial, model):        
    if model.lower() == "cnn":
        model_params = dict(
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            lr=trial.suggest_float("lr", 1e-6, 1e-2, step=0.1),
            weight_decay=trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
            epochs=trial.suggest_int("epochs", 100, 500, step=100),
            dropout=trial.suggest_float("dropout", 0.0, 0.3),
            kernel_size=trial.suggest_categorical("kernel_size", [3, 5, 7]),
            pooling=trial.suggest_categorical("pooling", ["max", "avg"]),
            pool_kernel=trial.suggest_categorical("pool_kernel", [2, 3]),
            batch_norm=trial.suggest_categorical("batch_norm", [True, False]),
            hidden_dim_fc=trial.suggest_categorical("hidden_dim_fc", [32, 64, 128]),
            num_layers = trial.suggest_int("num_layers", low=2, high=4, step=1)
        )
    elif model.lower() in ["lstm", "gru"]:
        model_params = dict(
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-2, 1e-1, log=True),
            epochs=trial.suggest_int("epochs", 100, 500, step=100),
            hidden_dim=trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            num_layers = trial.suggest_int("num_layers", low=1, high=4, step=1),
            dropout=trial.suggest_float("dropout", 0.0, 0.3),
            bias= True,
            batch_first=True
        )
    elif model.lower() == "transformer":
        hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64])
        valid_nheads = [n for n in [2, 3, 4, 5, 6, 7, 8] if hidden_dim % n == 0]
        model_params = dict(
            batch_size = trial.suggest_categorical("batch_size", [16, 32]),
            lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-2, 1e-1, log=True),
            epochs=trial.suggest_int("epochs", 100, 500, step=100),
            hidden_dim=hidden_dim,
            num_heads=trial.suggest_categorical("num_heads", valid_nheads),
            num_layers = trial.suggest_int("num_layers", low=1, high=4, step=1),
            dropout=trial.suggest_float("dropout", 0.0, 0.3),
            max_seq_len=1023
        )
    elif model.lower() == "mamba":
        model_params = dict(
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-2, 1e-1, log=True),
            epochs=trial.suggest_int("epochs", 100, 500, step=100),
            d_state=trial.suggest_categorical("d_state", [16, 32, 64, 96, 128]),
            expand=trial.suggest_int("expand", 1, 4),
            d_conv=trial.suggest_int("d_conv", 3, 6),
            dropout=trial.suggest_float("dropout", 0.1, 0.3),
            pooling=trial.suggest_categorical("pooling", ["max", "avg", "last"])
        )
    else:
        raise ValueError(f"Unknown model name: {model}")

    return model_params

def objective_ml_model(trial, X, y, model_name, cv=False, folds=None, 
                       search_params=None, score_metric="auroc", save_path=None):    

    valid_metrics = [
        "accuracy",
        "f1",
        "precision",
        "sensitivity",
        "specificity",
        "auroc",
        "auprc",
    ]
    assert score_metric in valid_metrics, f"Invalid score_metric: {score_metric}. Must be one of: {valid_metrics}"
    
    model_dict = {
        "xgb": xgb,
        "lgbm": LGBMClassifier,
        "svm": SVC,
        "rf": RandomForestClassifier,
        "gbdt": GradientBoostingClassifier,
        "logreg": LogisticRegression,
        "knn": KNeighborsClassifier,
        "pca_lda": LinearDiscriminantAnalysis,
        "pls_da": PLSRegression
    }
      
    param = {}
    if search_params is not None:
        for key, val in search_params.items():
            if isinstance(val, dict):
                param[key] = customize_param(trial, search_params, key)
            else: 
                param[key] = val   
         
    else:
        param = suggest_ml_hyperparams(trial, model_name)
          
    model_cls = model_dict[model_name]    
    
    if save_path is None:
        save_path = os.getcwd()
    result_path = f"{save_path}/result"
        
    all_metrics  = []    

    if cv is not False:
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_train, y_train = X[train_idx], y[train_idx]
            X_valid, y_valid = X[val_idx], y[val_idx]
        
            if model_name == 'xgb':    
                dtrain = model_cls.DMatrix(X_train, label=y_train)
                dvalid = model_cls.DMatrix(X_valid, label=y_valid)
                num_round = trial.suggest_int("n_estimators", 50, 300)
                model = model_cls.train(param, dtrain, num_boost_round=num_round)
                prods = model.predict(dvalid)
                preds = (prods > 0.5).astype(int)
            elif model_name == 'svm':
                model = model_cls(**param, probability=True)
                model.fit(X_train, y_train)
                preds = model.predict(X_valid)
                prods = model.predict_proba(X_valid)[:, 1]                
            # ===== 新增：PCA-LDA =====
            elif model_name == "pca_lda":
                # 构建 Pipeline
                lda_kwargs = {}
                if "lda__solver" in param:
                    lda_kwargs["solver"] = param["lda__solver"]
                if param.get("lda__solver") == "lsqr":
                    lda_kwargs["shrinkage"] = param.get("lda__shrinkage", None)

                model = Pipeline([
                    ("pca", PCA(n_components=param["pca__n_components"],svd_solver="full", random_state=param["random_state"])),
                    ("lda", model_cls(**lda_kwargs))
                ])
                model.fit(X_train, y_train)
                preds = model.predict(X_valid)
                prods = model.predict_proba(X_valid)[:, 1]

            elif model_name == "pls_da":
                model = model_cls(n_components=param["pls__n_components"])
                model.fit(X_train, y_train)
                y_pred_cont = model.predict(X_valid)  

                if y_pred_cont.ndim > 1 and y_pred_cont.shape[1] > 1:
                    preds = np.argmax(y_pred_cont, axis=1)
                    prods = y_pred_cont 
                else:
                    prods = y_pred_cont.ravel()  
                    preds = (prods > 0.5).astype(int)

            else:
                model = model_cls(**param)
                model.fit(X_train, y_train)
                preds = model.predict(X_valid)
                prods = model.predict_proba(X_valid)[:, 1]

            scores.append(defined_metric_score(score_metric, y_valid, preds, prods))
            metrics = evaluate(y_valid, preds, prods)
            fold_result = {
                "fold": fold_idx,
                "y_true": y_valid,
                "y_pred_lab": preds,
                "y_pred_prc": prods, **metrics}
            all_metrics.append(fold_result)
                
        df = pd.DataFrame(all_metrics)
        
        metric_cols = df.iloc[:, 8:]
        col_means = metric_cols.mean(axis=0)  
        col_stds = metric_cols.std(axis=0)

        df.loc[len(df)] = ["Mean"] + ["-"] * 7 + col_means.tolist()  

        mean_std_row = ["Mean ± Std"] + ["-"] * 7
        for mean, std in zip(col_means, col_stds):
            mean_std_row.append(f"{mean:.4f}±{std:.4f}")
        df.loc[len(df)] = mean_std_row
        
        cv_path = f"{result_path}/cv_result"
        os.makedirs(cv_path, exist_ok=True)
        df.to_csv(f"{cv_path}/{model_name}_trial_{trial.number}_crossval_results.csv", index=False)  
    
        return np.mean(scores)

    else:
        if model_name == 'xgb':    
            dtrain = model_cls.DMatrix(X, label=y)
            num_round = param.pop("n_estimators", 100)
            model = model_cls.train(param, dtrain, num_boost_round=num_round)
            prods = model.predict(dtrain)
            preds = (prods > 0.5).astype(int)
        elif model_name == 'svm':
            model = model_cls(**param, probability=True)
            model.fit(X, y)
            preds = model.predict(X)
            prods = model.predict_proba(X)[:, 1]   
        elif model_name == "pca_lda":
            lda_kwargs = {}
            if "lda__solver" in param:
                lda_kwargs["solver"] = param["lda__solver"]
            if param.get("lda__solver") == "lsqr":
                lda_kwargs["shrinkage"] = param.get("lda__shrinkage", None)
            model = Pipeline([
                ("pca", PCA(n_components=param["pca__n_components"],svd_solver="full",random_state=param["random_state"])),
                ("lda", model_cls(**lda_kwargs))
            ])
            model.fit(X, y)
            preds = model.predict(X)
            prods = model.predict_proba(X)[:, 1]
        elif model_name == "pls_da":
            model = model_cls(n_components=param["pls__n_components"])
            model.fit(X, y)
            y_pred_cont = model.predict(X)  
            if y_pred_cont.ndim > 1 and y_pred_cont.shape[1] > 1:
                preds = np.argmax(y_pred_cont, axis=1)
                prods = y_pred_cont 
            else:
                prods = y_pred_cont.ravel() 
                preds = (prods > 0.5).astype(int)
        else:
            model = model_cls(**param)
            model.fit(X, y)
            preds = model.predict(X)
            prods = model.predict_proba(X)[:, 1]
        
        save_model_path = f"{save_path}/model"
        os.makedirs(save_model_path, exist_ok=True)          
        save_model_file = f"{save_model_path}/best_{model_name}_model.pth" 
        save_ml_model(model, save_model_file)
        
        score = defined_metric_score(score_metric, y, preds, prods)
    
        metrics = evaluate(y, preds, prods)
        all_metrics = {
            "y_true": y.tolist(),
            "y_pred_lab": preds.tolist(),
            "y_pred_prc": prods.tolist(), **metrics}

        df = pd.DataFrame([all_metrics])
        df.to_csv(f"{result_path}/best_{model_name}_on_train_results.csv", index=False)  
    
        return score, model
    

def objective_dl_model(trial, X, y, model_name,
                       cv=False, folds=None, 
                       search_params=None,
                       score_metric="auroc",
                       save_path=None, 
                       device="cpu"):
  
    valid_metrics = [
        "accuracy",
        "f1",
        "precision",
        "sensitivity",
        "specificity",
        "auroc",
        "auprc",
    ]
    assert score_metric in valid_metrics, f"Invalid score_metric: {score_metric}. Must be one of: {valid_metrics}"

    model_dict = {
        "cnn": CNN1D,
        "lstm": LSTM1D,
        "gru": GRU1D,
        "transformer": Transformer1D,
        "mamba": Mamba1D
    }
    
    model_class = model_dict[model_name] 
            
    if save_path is None:
        save_path = os.getcwd()
    result_path = os.path.join(save_path, "result")
    os.makedirs(result_path, exist_ok=True)

    param = {}
    if search_params is not None:
        for key, val in search_params.items():
            if isinstance(val, dict): 
                    param[key] = customize_param(trial, search_params, key)
            else:  
                param[key] = val   

        param.setdefault("lr", 1e-3)
        param.setdefault("weight_decay", 1e-2)
        param.setdefault("epochs", 50)
        param.setdefault("batch_size", 32)
    else:
        param = suggest_dl_hyperparams(trial, model_name)

    if model_name.lower() in ["lstm" , "gru", "mamba", "transformer"]:
        input_dim = 1
    else:
        input_dim = X.shape[1] 
           
    if cv is not False:
        all_metrics = []
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            
            X_train, X_val = torch.FloatTensor(X[train_idx]), torch.FloatTensor(X[val_idx])
            y_train, y_val = torch.LongTensor(y[train_idx]), torch.LongTensor(y[val_idx])                   
            
            train_loader = DataLoader(TensorDataset(X_train, y_train), 
                                      batch_size=param["batch_size"], 
                                      shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val), 
                                    batch_size=param["batch_size"])
            model = model_class(input_dim=input_dim, **param).to(device)
            result = train_pytorch_model(model, train_loader, val_loader, 
                                         epochs=param["epochs"], 
                                         lr=param["lr"], 
                                         weight_decay=param["weight_decay"],
                                         device=device)
            scores.append(result[score_metric]) 
            all_metrics.append({"fold": fold_idx, **result})
            
        df = pd.DataFrame(all_metrics)
        
        col_means = df.iloc[:, 8:].mean(axis=0) 

        df.loc[len(df)] = ["Mean"] + ["-"] * 7 + col_means.tolist()  
        
        if save_path is None:
            save_path = os.getcwd()
        result_path = f"{save_path}/result"
        os.makedirs(result_path, exist_ok=True)
        df.to_csv(f"{result_path}/{model_name}_crossval_results.csv", index=False)
        return np.mean(scores)              
    
    else:
        X_train = torch.FloatTensor(X)
        y_train = torch.LongTensor(y)
  
        train_loader = DataLoader(TensorDataset(X_train, y_train), 
                               batch_size=param["batch_size"], 
                               shuffle=True)
        model = model_class(input_dim=input_dim, **param)
        save_model_path = f"{save_path}/model"
        os.makedirs(save_model_path, exist_ok=True)          
        save_model_file = f"{save_model_path}/best_{model_name}_model.pth"
        result = train_pytorch_model(model, train_loader, val_loader=None, 
                                     epochs=param["epochs"], lr=param["lr"], 
                                     device=device,save_model=save_model_file)

        df = pd.DataFrame([result])
        df.to_csv(f"{result_path}/best_{model_name}_on_full_train_results.csv", index=False)
        return result[score_metric], model         