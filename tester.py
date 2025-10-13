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
    metrics = evaluate(all_labels, all_preds, all_probs)
    all_results = {
        "y_true": all_labels,
        "y_pred_lab": all_preds,
        "y_pred_prc": all_probs, **metrics}
    
    if save_path is None:
        save_path = os.getcwd()
    os.makedirs(save_path, exist_ok=True)
            
    df = pd.DataFrame([all_results])
    df.to_csv(f"{save_path}/result/best_{model_name}_on_test_results.csv", index=False)
            
    return all_results[score_metric]

def test_ml_model(model, model_name, X_test, y_test, score_metric, save_path=None):
    if model_name == 'xgb':    
        dtest = xgb.DMatrix(X_test, y_test)   
        prods = model.predict(dtest)
        preds = (prods > 0.5).astype(int)
    elif model_name == 'pls_da':  
        y_pred_cont = model.predict(X_test) 

        if y_pred_cont.ndim > 1 and y_pred_cont.shape[1] > 1:
            preds = np.argmax(y_pred_cont, axis=1)
            prods = y_pred_cont 
        else:
            prods = y_pred_cont.ravel() 
            preds = (prods > 0.5).astype(int)                            
    else:
        preds = model.predict(X_test)
        prods = model.predict_proba(X_test)[:, 1]
                    
    metrics = evaluate(y_test, preds, prods)
    all_results = {
        "y_true": y_test.tolist(),
        "y_pred_lab": preds.tolist(),
        "y_pred_prc": prods.tolist(), **metrics}

    if save_path is None:
        save_path = os.getcwd()
    os.makedirs(save_path, exist_ok=True) 
    
    df = pd.DataFrame([all_results])
    df.to_csv(f"{save_path}/result/best_{model_name}_on_test_results.csv", index=False)
    
    return all_results[score_metric]