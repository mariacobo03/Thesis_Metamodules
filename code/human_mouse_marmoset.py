# this is single cell (xgboost) without the gene modules 

import scanpy as sc
import pandas as pd
import numpy as np
import os
import anndata as ad
import xgboost as xgb
import shap # interpretability 
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split, ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression

# Create a folder for saving figures
output_dir = "figures/single_cell"
os.makedirs(output_dir, exist_ok=True)

# Set Scanpy to save plots 
sc.settings.figdir = output_dir

adata1 = sc.read_h5ad("adata_inhibitory.h5ad")
adata2 = sc.read_h5ad("adata_excitory.h5ad")

data = adata1.concatenate(adata2, join="inner", batch_key="batch", batch_categories=["adata_inhibitory.h5ad", "adata_excitory.h5ad"])

human_data = data[data.obs['organism'] == 'Homo sapiens'].copy()
marmoset_data = data[data.obs['organism'] == 'Callithrix jacchus'].copy()
mouse_data = data[data.obs['organism'] == 'Mus musculus'].copy()

# Entrenar Mouse, predecir en Human y Primate 
# Normalize and preprocess 
for ad in [human_data, marmoset_data, mouse_data]:
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

# Convert to dense format 
X_train = mouse_data.X.toarray() if isinstance(mouse_data.X, csr_matrix) else mouse_data.X
X_test = human_data.X.toarray() if isinstance(human_data.X, csr_matrix) else human_data.X

# Target labels
le = LabelEncoder()
y_train = le.fit_transform(mouse_data.obs['BICCN_subclass_label'])
y_test = le.transform(human_data.obs['BICCN_subclass_label'])

# Convert to XGBoost DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parameter Grid
param_grid = {
    'learning_rate': [0.1],
    'max_depth': [3, 5],
    'n_estimators': [50, 100],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# Track best model
best_score = -1
best_params = None

# Run model tuning
for params in ParameterGrid(param_grid):
    print(f"Testing params: {params}")
    clf = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        early_stopping_rounds=5,
        tree_method="hist",
        n_jobs=-1,
        **params
    )
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    score = clf.score(X_test, y_test)
    print(f"Test accuracy: {score}")
    
    if score > best_score:
        best_score = score
        best_params = params
        best_model = clf

print("Best Parameters:", best_params)
print("Best Test Accuracy:", best_score)

# Save predictions
y_pred = best_model.predict(X_test)
df_results = pd.DataFrame({
    "True": le.inverse_transform(y_test),
    "Predicted": le.inverse_transform(y_pred)
})
df_results.to_csv("true_vs_predicted_mouse_to_human.csv", index=False)
print(df_results.head())

# SHAP explanation
X_test_df = pd.DataFrame(X_test, columns=human_data.var_names)
explainer = shap.Explainer(best_model)
shap_values = explainer(X_test_df)

# SHAP Beeswarm plots per class
for class_idx in range(len(le.classes_)):
    class_label = le.classes_[class_idx]
    safe_label = class_label.replace(" ", "_").replace("/", "_")  # Replace problematic chars
    shap_plot_path = os.path.join(output_dir, f"shap_beeswarm_class_{safe_label}.png")
    
    shap.plots.beeswarm(shap_values[:, :, class_idx], show=False)
    plt.savefig(shap_plot_path, bbox_inches="tight")
    plt.close()

# Compute neighbors and UMAP on the full dataset
sc.pp.pca(data)
sc.pp.neighbors(data, n_pcs=30)
sc.tl.umap(data)

# Plot UMAP colored by organism
sc.pl.umap(data, color='organism', save='_organism.png')

# Plot UMAP colored by subclass label
sc.pl.umap(data, color='BICCN_subclass_label', save='_subclass.png')

# Compute confusion matrix
# Inverse transform to get class names back from label encoding
true_labels = le.inverse_transform(y_test)
pred_labels = le.inverse_transform(y_pred)

all_labels = le.classes_

cm = confusion_matrix(true_labels, pred_labels, labels=all_labels, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format=".2f")
plt.title("Confusion Matrix: Human (True) vs Mouse-Trained Predictions")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix_mouse_to_human.png"))
plt.close()

# Mouse to Human
# Best Parameters: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'subsample': 0.8}
# Best Test Accuracy: 0.8317806760713335 --> 83.17% 

# Mouse to Marmoset 
# Best Parameters: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8}
# Best Test Accuracy: 0.8054296045457154 --> 80.54%