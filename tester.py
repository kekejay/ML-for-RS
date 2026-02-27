import torch
import xgboost as xgb
import os
import pandas as pd 
from utils import evaluate
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np

def test_dl_model(model, model_name, score_metric, X_test, y_test, device="cpu", save_path=None):
    
    model.to(device)
    model.eval()
    cm_keys = ['tn', 'fp', 'fn', 'tp']
    score_keys = ['accuracy', 'f1', 'precision', 'sensitivity', 'specificity', 'auroc', 'auprc']

    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
                             batch_size=32)
            
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_batch = y_batch.view(-1).float()
            X_batch = X_batch.to(device)
            outputs = model(X_batch).cpu()
            all_probs.extend(outputs.numpy())
            all_preds.extend((outputs > 0.5).int().numpy())
            all_labels.extend(y_batch.numpy())
    result = evaluate(all_labels, all_preds, all_probs)
    
    metrics = {**{k: result[k] for k in score_keys if k in result}}

    if save_path is None:
        save_path = os.getcwd()
    result_path = f"{save_path}/result/{model_name}"
    os.makedirs(result_path, exist_ok=True)
            
    df = pd.DataFrame([metrics])
    df.to_csv(f"{result_path}/best_model_on_test_results.csv", index=False)
       
    return metrics[score_metric]

def test_ml_model(model, model_name, X_test, y_test, score_metric, save_path=None):
    cm_keys = ['tn', 'fp', 'fn', 'tp']
    score_keys = ['accuracy', 'f1', 'precision', 'sensitivity', 'specificity', 'auroc', 'auprc']

    if model_name == 'xgb':    
        dtest = xgb.DMatrix(X_test, y_test)   
        prods = model.predict(dtest)
        preds = (prods > 0.5).astype(int)
    elif model_name == 'pls_da':
        #yb_test= pd.get_dummies(y_test)     
        y_pred_cont = model.predict(X_test) 

        if y_pred_cont.ndim > 1 and y_pred_cont.shape[1] > 1:
            preds = np.argmax(y_pred_cont, axis=1)
            prods = y_pred_cont
        else:
            prods = y_pred_cont.ravel()  # shape = (n_samples,)
            preds = (prods > 0.5).astype(int)

                            
    else:
        preds = model.predict(X_test)
        prods = model.predict_proba(X_test)[:, 1]
                    
    result = evaluate(y_test, preds, prods)
    metrics = {**{k: result[k] for k in score_keys if k in result}}
    
    if save_path is None:
        save_path = os.getcwd() 
    
    result_path = f"{save_path}/result/{model_name}"
    os.makedirs(result_path, exist_ok=True)
    
    df = pd.DataFrame([metrics])
    df.to_csv(f"{result_path}/best_model_on_test_results.csv", index=False)
    
    return metrics[score_metric]