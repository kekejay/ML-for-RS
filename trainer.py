import torch
import torch.nn as nn
from utils import evaluate, save_dl_model

def train_pytorch_model(
    model, train_loader, val_loader=None, save_model=None,
    epochs=50, lr=1e-3, weight_decay=0.01, pos_weight=None, device="cpu",
    switch_mode="plateau", switch_epoch=None, sgd_lr=1e-6, sgd_momentum=0.9, 
    sgd_weight_decay=None, plateau_patience=5, plateau_min_delta=5e-4,
    early_stopping_patience=15 
):
    """支持全量训练模式(val_loader=None)和常规训练验证模式"""
    import torch 
    import torch.nn as nn  
    from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau 
    from sklearn.metrics import roc_curve
    import numpy as np 
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.optim.lr_scheduler import LambdaLR    
    import copy 
    
    model_class_name = type(model).__name__
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    if "LSTM" in model_class_name:
        warmup_epochs = 10 
        def warmup_lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1, warmup_epochs)) 
            return 1.0 
              
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)      
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
        scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
    
    elif "CNN" in model_class_name:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=sgd_momentum)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1) 
    else:
        if sgd_lr is None:
            sgd_lr = lr * 0.1
        if sgd_weight_decay is None:
            sgd_weight_decay = weight_decay
        if switch_epoch is None:
            switch_epoch = max(1, int(0.4 * epochs))
            
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-8)


    model.to(device)

    def _evaluate_on_loader(eval_loader):
        model.eval()
        losses, y_true, y_prob, y_pred = [], [], [], []
        with torch.no_grad():
            for Xb, yb in eval_loader:
                Xb = Xb.to(device); yb = yb.to(device).float().view(-1)
                out = model(Xb).view(-1)
                loss = criterion(out, yb); losses.append(loss.item())
                prob = torch.sigmoid(out).detach().cpu().numpy()
                y_prob.extend(prob)
                y_pred.extend((prob > 0.5).astype(int))
                y_true.extend(yb.detach().cpu().numpy())
        mean_loss = float(np.mean(losses)) if len(losses) else np.nan
        metrics = evaluate(y_true, y_pred, y_prob) 
        return mean_loss, metrics, y_true, y_pred, y_prob
        
    history = {"train_loss": [], "val_loss": [], "train_auroc": [], "val_auroc": []}
    
    no_improve_cnt = 0
    best_val_loss_for_plateau = float("inf")

    best_val_loss_global = float("inf")
    best_model_state = None
    early_stop_cnt = 0
    
    for epoch in range(epochs):
        model.train()
        train_losses, y_true_tr, y_prob_tr, y_pred_tr = [], [], [], []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()
            y_batch = y_batch.view(-1).float()
            optimizer.zero_grad()
            outputs = model(X_batch).view(-1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

            prob = torch.sigmoid(outputs).detach().cpu().numpy()
            y_prob_tr.extend(prob)
            y_pred_tr.extend((prob > 0.5).astype(int))
            y_true_tr.extend(y_batch.detach().cpu().numpy())
            
        train_metrics = evaluate(y_true_tr, y_pred_tr, y_prob_tr)
        history["train_loss"].append(float(np.mean(train_losses)))
        history["train_auroc"].append(train_metrics.get("auroc", np.nan))


        if val_loader is not None:
            mean_val_loss, metrics, all_labels, all_preds, all_probs = _evaluate_on_loader(val_loader)
            history["val_loss"].append(mean_val_loss)
            history["val_auroc"].append(metrics.get("auroc", np.nan))
            
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(mean_val_loss)
            elif isinstance(scheduler, LambdaLR): 
                scheduler.step()
                if epoch >= warmup_epochs:
                     cosine_scheduler.step()
            else:
                scheduler.step()

            if mean_val_loss < best_val_loss_global:
                best_val_loss_global = mean_val_loss
                best_model_state = copy.deepcopy(model.state_dict()) 
                early_stop_cnt = 0 
            else:
                early_stop_cnt += 1
            
            if early_stop_cnt >= early_stopping_patience:
                print(f"[Early Stopping] Triggered at epoch {epoch+1}. Best Loss: {best_val_loss_global:.4f}")
                break
            # ================================================================

            if "CNN" not in model_class_name and "LSTM" not in model_class_name:   
                if mean_val_loss < best_val_loss_for_plateau - plateau_min_delta:
                    best_val_loss_for_plateau = mean_val_loss
                    no_improve_cnt = 0
                else:
                    no_improve_cnt += 1
                
                should_switch = False
                
                if switch_mode == "plateau" and no_improve_cnt >= plateau_patience:
                    print(f"\n[Info] Validation loss plateaued for {no_improve_cnt} epochs.")
                    should_switch = True
                    
                if switch_mode == "epoch" and switch_epoch is not None and (epoch + 1) == switch_epoch:
                    print(f"\n[Info] Reached switch epoch {switch_epoch}.")
                    should_switch = True

                if should_switch:
                    print(f"[Switch] Switching optimizer from AdamW to SGD (LR={sgd_lr})")
                    
                    if best_model_state is not None:
                         model.load_state_dict(best_model_state)

                    optimizer = torch.optim.SGD(
                        model.parameters(),
                        lr=sgd_lr, momentum=sgd_momentum,
                        weight_decay=sgd_weight_decay if sgd_weight_decay is not None else weight_decay
                    )
                    
                    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-8)
                    
                    no_improve_cnt = 0
                    switch_mode = "done" 
                    early_stop_cnt = 0 

        else:
            save_dl_model(model, save_model)
                
            model.eval()
            all_preds, all_probs, all_labels = [], [], []
            with torch.no_grad():
                for X_batch, y_batch in train_loader:
                    y_batch = y_batch.view(-1).float()
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch).cpu()

                    all_probs.extend(outputs.numpy())
                    all_preds.extend((outputs > 0.5).int().numpy())
                    all_labels.extend(y_batch.numpy())

            metrics = evaluate(all_labels, all_preds, all_probs)
    
    if val_loader is not None and best_model_state is not None:
        print(f"Loading best model state from epoch with Loss: {best_val_loss_global:.4f}")
        model.load_state_dict(best_model_state)
        
        if save_model:
            save_dl_model(model, save_model)

        _, metrics, all_labels, all_preds, all_probs = _evaluate_on_loader(val_loader)
    # ====================================================================

    return {
        "result": {
            "y_true": all_labels,
            "y_pred_lab": all_preds,
            "y_pred_prc": all_probs,
            **metrics,  
        },
        "histories": history  
    }
        