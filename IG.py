import torch
import numpy as np
import pandas as pd
from captum.attr import IntegratedGradients

def compute_ig_with_feature_ranking(model, X_tensor, feature_names,
                                    batch_size=64, n_steps=50, device="cpu",
                                    internal_batch_size=None):

    model.to(device)
    model.eval()
    
    X_tensor = X_tensor.to(device)

    ig = IntegratedGradients(model)
    all_attr = []
    all_delta = []

    for i in range(0, X_tensor.shape[0], batch_size):
        x_batch = X_tensor[i:i+batch_size]

        attr_batch, delta_batch = ig.attribute(
            x_batch,
            target=None,
            return_convergence_delta=True,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size
        )

        all_attr.append(attr_batch.detach().cpu())
        all_delta.append(delta_batch.detach().cpu())

        torch.cuda.empty_cache()

    attr_all = torch.cat(all_attr).numpy()
    delta_all = torch.cat(all_delta).numpy()

    mean_abs_attr = np.mean(np.abs(attr_all), axis=0) 
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_attr
    })

    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    return importance_df, attr_all, delta_all