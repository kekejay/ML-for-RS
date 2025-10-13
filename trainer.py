import torch
import torch.nn as nn
from utils import evaluate, save_dl_model

def train_pytorch_model(model, train_loader, val_loader=None, save_model=None,
                        epochs=50, lr=1e-3, weight_decay=0.01, device="cpu"):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()
            y_batch = y_batch.view(-1).float()
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    if val_loader is None:
        save_dl_model(model,save_model)
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
        return {
            "y_true": all_labels,
            "y_pred_lab": all_preds,
            "y_pred_prc": all_probs,**metrics
            }
            
    else:
        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_batch = y_batch.view(-1).float()
                X_batch = X_batch.to(device)
                outputs = model(X_batch).cpu()
                all_probs.extend(outputs.numpy())
                all_preds.extend((outputs > 0.5).int().numpy())
                all_labels.extend(y_batch.numpy())

        metrics = evaluate(all_labels, all_preds, all_probs)
        return {
            "y_true": all_labels,
            "y_pred_lab": all_preds,
            "y_pred_prc": all_probs, **metrics
            }

        