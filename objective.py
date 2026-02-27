import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np
import os
import pandas as pd
from utils import evaluate, save_ml_model
import torch
from torch.utils.data import DataLoader, TensorDataset
from trainer import train_pytorch_model, aggregate_cv_results
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
        # 固定值或没定义时直接返回
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
            lr=trial.suggest_float("lr", 1e-6, 1e-4, log=True),#原来1e-5~1e-3
            weight_decay=trial.suggest_float("weight_decay", 1e-2, 1e-1, log=True), #原来1e-4, 1e-1
            epochs=trial.suggest_int("epochs", 50, 200, step=10),
            dropout=trial.suggest_float("dropout", 0.5, 0.7), #原来0-0.3
            kernel_size=trial.suggest_categorical("kernel_size", [3, 5, 7]),
            pooling=trial.suggest_categorical("pooling", ["max", "avg"]),
            pool_kernel=trial.suggest_categorical("pool_kernel", [2, 3]),
            batch_norm=trial.suggest_categorical("batch_norm", [True, False]),
            hidden_dim_fc=trial.suggest_categorical("hidden_dim_fc", [32, 64, 128]),
            num_layers = trial.suggest_int("num_layers", low=2, high=4, step=1)
        )
    elif model.lower() in ["lstm", "gru"]:
        model_params = dict(
            batch_size = trial.suggest_categorical("batch_size", [16,32, 64]),
            lr=trial.suggest_float("lr", 1e-6, 1e-4, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-2, 1e-1, log=True),
            epochs=trial.suggest_int("epochs", 50, 200, step=10),
            hidden_dim=trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            num_layers = trial.suggest_int("num_layers", low=1, high=3, step=1),
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
            epochs=trial.suggest_int("epochs", 50, 200, step=10),
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
            epochs=trial.suggest_int("epochs", 50, 200, step=10),
            d_state=trial.suggest_categorical("d_state", [16, 32, 64, 96, 128]),
            expand=trial.suggest_int("expand", 1, 4),
            d_conv=trial.suggest_int("d_conv", 3, 6),
            dropout=trial.suggest_float("dropout", 0.1, 0.3),
            pooling=trial.suggest_categorical("pooling", ["max", "avg", "last"])
        )
    else:
        raise ValueError(f"Unknown model name: {model}")

    return model_params

def objective_ml_model(trial, X, y, model_name, IsTrial=False, folds=None, 
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
      
    #获取参数
    param = {}
    if search_params is not None:
        for key, val in search_params.items():
            if isinstance(val, dict):  # 需要采样的参数
                param[key] = customize_param(trial, search_params, key)
            else:  # 固定值，直接赋值
                param[key] = val   
         
    else:#默认搜索
        param = suggest_ml_hyperparams(trial, model_name)
          
    # 获取模型类
    model_cls = model_dict[model_name]    
    
    if save_path is None:
        save_path = os.getcwd()

    cm_keys = ['tn', 'fp', 'fn', 'tp']
    score_keys = ['accuracy', 'f1', 'precision', 'sensitivity', 'specificity', 'auroc', 'auprc']
          
    # ------------------ 交叉验证阶段 ------------------ #
    if IsTrial is not False:
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

            metrics = evaluate(y_valid, preds, prods)
            scores.append(metrics[score_metric])
           
        return np.mean(scores)

    # ------------------ 最终模型训练阶段 ------------------ #
    else:
        all_metrics = []
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

            metrics = evaluate(y_valid, preds, prods)
            fold_result = {"fold": fold_idx, **{k: metrics[k] for k in score_keys if k in metrics}}
            all_metrics.append(fold_result)
                
        df_metrics = pd.DataFrame(all_metrics)
        
        #metric_cols = df_metrics.iloc[:, 8:]
        col_means = df_metrics.iloc[:, 1:].mean(axis=0)  
        col_stds = df_metrics.iloc[:, 1:].std(axis=0)

        mean_std_row = ["Mean ± Std"]
        for mean, std in zip(col_means, col_stds):
            mean_std_row.append(f"{mean:.4f}±{std:.4f}")
        df_metrics.loc[len(df_metrics)] = mean_std_row
        
        result_path = f"{save_path}/result/{model_name}"
        os.makedirs(result_path, exist_ok=True)
        df_metrics.to_csv(f"{result_path}/crossval_results.csv", index=False)              
        
               
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
        
        save_model_path = f"{save_path}/model/{model_name}"
        os.makedirs(save_model_path, exist_ok=True)          
        save_model_file = f"{save_model_path}/best_model.pth" 
        save_ml_model(model, save_model_file)
        
        result = evaluate(y, preds, prods)
        score = result[score_metric]
        metrics = {**{k: metrics[k] for k in score_keys if k in result}}
        df = pd.DataFrame([metrics])
        df.to_csv(f"{result_path}/best_model_on_full_train_results.csv", index=False)
        
        return score, model
    

def objective_dl_model(trial, X, y, model_name,
                       IsTrial=False, folds=None, 
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
    
    # 获取模型类
    model_class = model_dict[model_name] 
            
    if save_path is None:
        save_path = os.getcwd()

    cm_keys = ['y_true','y_pred_lab','y_pred_prc','tn', 'fp', 'fn', 'tp']
    score_keys = ['accuracy', 'f1', 'precision', 'sensitivity', 'specificity', 'auroc', 'auprc']


    param = {}
    if search_params is not None:
        for key, val in search_params.items():
            if isinstance(val, dict):  # 需要采样的参数
                    param[key] = customize_param(trial, search_params, key)
            else:  # 固定值，直接赋值
                param[key] = val   
        # 设置默认值（如果缺失）
        param.setdefault("lr", 1e-3)
        param.setdefault("weight_decay", 1e-2)
        param.setdefault("epochs", 50)
        param.setdefault("batch_size", 32)
    else:
        param = suggest_dl_hyperparams(trial, model_name)

    if model_name.lower() in ["gru","lstm"]:
        input_dim = 1
        SGD_lr = 1e-6  #lstm,gru
    elif model_name.lower() in ["mamba", "transformer"]:
        input_dim = 1
        SGD_lr = 5e-6 #transformer
    else:
        SGD_lr = 1e-6
        input_dim = X.shape[1] 

          
    # 保存每折结果
    if IsTrial is not False:
        #交叉验证
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
                                         device=device,
                                         sgd_lr=SGD_lr,
                                         pos_weight=torch.tensor(np.count_nonzero(y_val == 0)/np.count_nonzero(y_val == 1)))
            metrics = result["result"]
            scores.append(metrics[score_metric])  # 可替换为其他指标

        return np.mean(scores)              
    
    else:
        # 用最佳参数再次交叉验证，以获得所有评价指标
        all_histories = []
        all_metrics = []
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
                                         device=device,
                                         sgd_lr=SGD_lr,
                                         pos_weight=torch.tensor(np.count_nonzero(y_val == 0)/np.count_nonzero(y_val == 1)))
            metrics = result["result"]
            histories = result["histories"]
            all_histories.append(histories)
            all_metrics.append({"fold": fold_idx, **{k: metrics[k] for k in score_keys if k in metrics}})       
        
        result_path = f"{save_path}/result/{model_name}"
        os.makedirs(result_path, exist_ok=True) 
    
        aggregate_cv_results(all_histories, save_dir=result_path)
        df_metrics = pd.DataFrame(all_metrics)   
        
        col_means = df_metrics.iloc[:, 1:].mean(axis=0)  
        col_stds = df_metrics.iloc[:, 1:].std(axis=0)

        mean_std_row = ["Mean±Std"]
        for mean, std in zip(col_means, col_stds):
            mean_std_row.append(f"{mean:.4f}±{std:.4f}")
        df_metrics.loc[len(df_metrics)] = mean_std_row
    
        df_metrics.to_csv(f"{result_path}/crossval_results.csv", index=False)
        # 全量数据模式：忽略交叉验证，使用全部数据训练
        X_train_all = torch.FloatTensor(X)
        y_train_all = torch.LongTensor(y)
  
        train_loader_all = DataLoader(TensorDataset(X_train_all, y_train_all), 
                               batch_size=param["batch_size"], 
                               shuffle=True)
        model = model_class(input_dim=input_dim, **param)
        save_model_path = f"{save_path}/model/{model_name}"
        os.makedirs(save_model_path, exist_ok=True)          
        save_model_file = f"{save_model_path}/best_model.pth"
        result = train_pytorch_model(model, train_loader_all, val_loader=None, 
                                     epochs=param["epochs"], lr=param["lr"],sgd_lr=SGD_lr, 
                                     device=device,save_model=save_model_file)
        metrics = {**{k: metrics[k] for k in score_keys if k in result["result"]}}
        
        df = pd.DataFrame([metrics])
        df.to_csv(f"{result_path}/best_model_on_full_train_results.csv", index=False)
        return metrics[score_metric], model         