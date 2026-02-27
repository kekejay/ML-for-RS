import joblib
import pandas as pd 
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split

def extract_class1_shap(shap_values):
    if isinstance(shap_values, shap.Explanation):
        if shap_values.values.ndim == 3:
            return shap.Explanation(
                values=shap_values.values[:, :, 1],
                base_values=shap_values.base_values[:, 1] if shap_values.base_values.ndim == 2 else shap_values.base_values,
                data=shap_values.data,
                feature_names=shap_values.feature_names
            )
        else:
            return shap_values
        
    elif isinstance(shap_values, list):
        return shap_values[1] 
    
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            return shap_values[:, :, 1] 
        elif shap_values.ndim == 2:
            return shap_values 
        else:
            raise ValueError(f"Unsupported numpy shape for SHAP values: {shap_values.shape}")
        
    else:
        raise ValueError("Unsupported SHAP format")


def get_shap_explainer(model, model_name, X_train_df):
    tree_models = ["xgb", "rf", "lgbm", "gbdt"]
    linear_models = ["logreg"]
    Kernel_models = ["knn"]
    
    if model_name in tree_models:
        return shap.TreeExplainer(model)
    elif model_name in linear_models or \
         (model_name == "svm" and hasattr(model, "kernel") and model.kernel == "linear"):
        masker = shap.maskers.Independent(X_train_df)
        return shap.LinearExplainer(model, masker=masker)
    elif model_name in Kernel_models or \
         (model_name == "svm" and hasattr(model, "kernel") and model.kernel == "rbf"):
        background = shap.kmeans(X_train_df, 100)
        return shap.KernelExplainer(model.predict_proba, background)
    else:
        background = shap.kmeans(X_train_df, 100)
        return shap.KernelExplainer(model.predict_proba, background)
    
    
def shap_analysis(
    model,
    model_name,
    X_train_df,
    X_test_df,
    y_train,
    y_test,
    save_dir="./shap_out",
    prefix="Glioma_cancer_new",
    nsamples=100,
    n_samples=500
):
    
    os.makedirs(save_dir, exist_ok=True)
    
    model_name_map = {
        "xgb": "XGBoost",
        "rf": "RF",
        "lgbm": "LGBM",
        "gbdt": "GBDT",
        "svm": "SVM",
        "knn": "KNN",
        "logreg": "Logreg"
    }
    
    print("📦 Prepare SHAP Explainer ...")
    explainer = get_shap_explainer(model, model_name, X_train_df)
    
    is_kernel = isinstance(explainer, shap.KernelExplainer)
    sample_args = {"nsamples": nsamples} if is_kernel else {}

    if is_kernel:
        print(f"\n✂️ 正在对测试集进行分层采样 (目标: {n_samples} 样本)...")
        X_train_sample, _, y_sample, _ = train_test_split(
            X_test_df, y_train, 
            train_size=n_samples, 
            stratify=y_train, 
            random_state=1
        )
        X_test_sample, _, y_sample, _ = train_test_split(
            X_test_df, y_test, 
            train_size=n_samples, 
            stratify=y_test, 
            random_state=1
        )
        X_test_sample = X_test_df.sample(n=500, random_state=1)
        X_train_sample.to_csv(f"{save_dir}/{prefix}/shap_train_{model_name}_sample.csv",index=False)
        X_test_sample.to_csv(f"{save_dir}/{prefix}/shap_test_{model_name}_sample.csv",index=False)
        
        base_values = explainer.expected_value
        joblib.dump(base_values, f"{save_dir}/{prefix}/{model_name}/shap_{model_name}_base_values.pkl")
        
        print("🔍 Calculate the SHAP value of the training set...")
        shap_values_train = explainer.shap_values(X_train_sample.values, **sample_args)
        print("🔍 Calculate the SHAP value of the test set ...")
        shap_values_test = explainer.shap_values(X_test_sample.values, **sample_args)
        
        shap_used_train = extract_class1_shap(shap_values_train)
        shap_used_test = extract_class1_shap(shap_values_test)
    
        shap_used_train_value = shap_used_train
        shap_used_test_value = shap_used_test
        
    else:
        X_train_sample = X_train_df
        X_test_sample = X_test_df
        
        print("🔍 Calculate the SHAP value of the training set ...")
        shap_values_train = explainer(X_train_df.values)
        print("🔍 Calculate the SHAP value of the test set ...")
        shap_values_test = explainer(X_test_df.values)

        shap_used_train = extract_class1_shap(shap_values_train)
        shap_used_test = extract_class1_shap(shap_values_test)
        
        shap_used_train_value = shap_used_train.values
        shap_used_test_value = shap_used_test.values

    pd.DataFrame(shap_used_train_value, columns=X_train_df.columns).to_csv(
                f"{save_dir}/{prefix}/shap_train_{model_name}.csv", index=False)
    pd.DataFrame(shap_used_test_value, columns=X_test_df.columns).to_csv(
                f"{save_dir}/{prefix}/shap_test_{model_name}.csv", index=False)
        
        
    for name, shap_vals, X in [("train", shap_used_train_value, X_train_sample),
                               ("test", shap_used_test_value, X_test_sample)]:
        plt.figure()
        shap.summary_plot(shap_vals, X, plot_type="bar", show=False)
        plt.title(f"{model_name_map[model_name]} - {name} - SHAP Bar")
        plt.tight_layout()
        #plt.show()
        plt.savefig(f"{save_dir}/{prefix}/shap_{model_name}_{name}_bar.pdf")
        plt.close()

        plt.figure()
        shap.summary_plot(shap_vals, X, plot_type="dot", show=False)
        plt.title(f"{model_name_map[model_name]} - {name} - SHAP Dot")
        plt.tight_layout()
        #plt.show()
        plt.savefig(f"{save_dir}/{prefix}/shap_{model_name}_{name}_dot.pdf")
        plt.close()

        print(f"✅ {name.upper()}The image has been saved: bar + dot")
    print("🎉The SHAP analysis is completed.")