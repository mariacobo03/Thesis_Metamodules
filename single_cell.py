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


# Create a folder for saving figures
output_dir = "resultados"
os.makedirs(output_dir, exist_ok=True)

# Set Scanpy to save plots 
sc.settings.figdir = output_dir

# Define dataset directory and file paths
# dataset_directory = "datasets/Human/"
# expr_matrix_file = os.path.join(dataset_directory, "exprMatrix.tsv.gz")
# meta_file = os.path.join(dataset_directory, "meta.tsv")

# Load expression matrix
# expr_matrix = pd.read_csv(expr_matrix_file, sep="\t", compression="gzip", header=0)

# Process gene names (extract part after the last "|")
# genes = expr_matrix.iloc[:, 0].str.split('|').str[-1]
# expr_matrix = expr_matrix.iloc[:, 1:]
# expr_matrix.index = genes

# Load metadata
# metadata = pd.read_csv(meta_file, sep="\t", index_col=0)
# assert metadata.index.equals(expr_matrix.columns), "Cells IDs must match between metadata and expression matrix."

# Create Anndata object
# Transpose so cells are rows and genes are columns 
# adata = ad.AnnData(X=expr_matrix.T)
adata = sc.read_h5ad("datasets/Human/adata_human.h5ad")

mat = adata.to_df().T  # genes as rows, cells as columns

meta = adata.obs.copy()

# Ensure adata.X is float
adata.X = adata.X.astype(float)

# Check for infinite values and replace them
adata.X[np.isinf(adata.X)] = np.nan  # Replace inf/-inf with NaN
adata.X = np.nan_to_num(adata.X, nan=0)  # Replace NaN with 0 or another value

# Quality Control
adata.obs["total_counts"] = adata.X.sum(axis=1)
adata.obs["n_genes_by_counts"] = (adata.X > 0).sum(axis=1)
adata.var["n_cells_by_counts"] = (adata.X > 0).sum(axis=0)

# Normalize and log-transform 
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)  # Avoid log(0) issues

# Feature Selection 
adata = adata[adata.obs["total_counts"] > 0, :]
sc.pp.filter_genes(adata, min_cells=1)
# sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)  # Reduce features
sc.pp.scale(adata)  # Standardization for PCA and clustering 
# sc.pl.highly_variable_genes(adata, save="highly_variable_genes.png")

# Convert data to sparse format
X = csr_matrix(adata.X)  # Sparse matrix
X = adata.X.toarray() if isinstance(adata.X, csr_matrix) else adata.X

# Dimensionality Reduction 
sc.tl.pca(adata)
# Plot variance in PCA 
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True, save="pca_variance.png")

# Plot Principal component and color by cluster 
sc.pl.pca(adata, color="subclass_label", save="pca_cluster.png")

# Replace NaN values in the target column with "unknown"
target_column = "subclass_label"
if adata.obs[target_column].dtype.name == "category":
    adata.obs[target_column] = adata.obs[target_column].cat.add_categories("unknown")
adata.obs[target_column] = adata.obs[target_column].fillna("unknown")

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(adata.obs[target_column])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=94, stratify=y)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


# Fit and score function for cross-validation with early stopping
def fit_and_score(estimator, X_train, X_test, y_train, y_test):
    """Fit the estimator on the train set and score it on both sets"""
    # Fit the model with early stopping
    estimator.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    # Evaluate performance on training and test datasets
    train_score = estimator.score(X_train, y_train)
    test_score = estimator.score(X_test, y_test)
    
    return estimator, train_score, test_score


param_grid = {
    'learning_rate': [0.1],
    'max_depth': [5],
    'n_estimators': [100],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# Estimator - XGBoost classifier with early stopping
clf = xgb.XGBClassifier(
    objective="multi:softmax", 
    num_class=len(np.unique(y)),
    eval_metric="mlogloss",  # Ensure correct evaluation metric
    tree_method="hist", 
    early_stopping_rounds=3,
    n_jobs=-1  # Use all available cores
)

# Cross-validation loop
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
# dictionary to store results for each loop
results = {}

# variables to track the best model
best_score = -1
best_params = None

for params in ParameterGrid(param_grid):
    print(f"Testing with params: {params}")
    
    clf.set_params(**params)
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train_cv, X_test_cv = X[train_idx], X[test_idx] # feature sets
        y_train_cv, y_test_cv = y[train_idx], y[test_idx] # labels
        # fits and evaluates
        # clone so there is no data leakage between folds 
        est, train_score, test_score = fit_and_score(clone(clf), X_train_cv, X_test_cv, y_train_cv, y_test_cv)
        
        # if current fold is not in results, initialize a list
        if fold not in results:
            results[fold] = []
        
        results[fold].append({
            "params": params,
            "train_score": train_score,
            "test_score": test_score,
            "best_iteration": est.best_iteration
        })

# Find best hyperparameters
for fold_results in results.values():
    for result in fold_results:
        if result['test_score'] > best_score:
            best_score = result['test_score']
            best_params = result['params']

print(f"Best Hyperparameters: {best_params}")
print(f"Best Test Score: {best_score}")


if best_params:
    print(f"Training final model with best parameters: {best_params}")

    final_params = best_params.copy()
    final_params.update({
        "objective": "multi:softmax",
        "num_class": len(np.unique(y)),
        "eval_metric": "mlogloss"
    })

    # Remove 'n_estimators' since xgb.train uses num_boost_round instead
    num_boost_round = final_params.pop("n_estimators")

    final_model = xgb.train(final_params, dtrain, num_boost_round=num_boost_round)

    predictions = final_model.predict(dtest)
    predictions = [round(value) for value in predictions]

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)

    # Refit clf (XGBClassifier) to enable direct predictions
    clf.set_params(**best_params)
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    y_pred = clf.predict(X_test)

    result_df = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
    result_csv_path = os.path.join(output_dir, 'true_predicted_human.csv')
    result_df.to_csv(result_csv_path, index=False)
    print(result_df)
else:
    print("No best paramenters found. No final model training")


# Convert X_test to a DataFrame with feature names
X_test_df = pd.DataFrame(X_test, columns=adata.var_names)

# Initialize SHAP explainer
explainer = shap.TreeExplainer(final_model)
explanation = explainer(X_test_df)

# Compute SHAP values
shap_values = explanation.values  # Returns a SHAP Explanation object

# SHAP Beeswarm Plots for all detected classes
num_classes = explanation.values.shape[-1]  # Dynamically get the number of classes

for class_idx in range(num_classes):
    shap_plot_path = os.path.join(output_dir, f"shap_beeswarm_class_{class_idx}_human.png")
    shap.plots.beeswarm(explanation[:, :, class_idx], show=False)
    plt.savefig(shap_plot_path, bbox_inches='tight')
    plt.close()

# Compute neighbors and UMAP on the full dataset
sc.pp.pca(adata)
sc.pp.neighbors(adata, n_pcs=30)
sc.tl.umap(adata)

# Plot UMAP colored by organism
umap_path_organism = os.path.join(output_dir, "umap_human.png")
sc.pl.umap(adata, color='organism', save=None, show=False)
plt.savefig(umap_path_organism, bbox_inches='tight')
plt.close()

# Plot UMAP colored by subclass label
umap_path_subclass = os.path.join(output_dir, "umap_subclass_human.png")
sc.pl.umap(adata, color='subclass_label', save=None, show=False)
plt.savefig(umap_path_subclass, bbox_inches='tight')
plt.close()

# Compute confusion matrix
# Inverse transform to get class names back from label encoding
true_labels = label_encoder.inverse_transform(y_test)
pred_labels = label_encoder.inverse_transform(y_pred)
all_labels = label_encoder.classes_

cm = confusion_matrix(true_labels, pred_labels, labels=all_labels, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format=".2f")
plt.title("Confusion Matrix: Human (True vs Predictions)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix_human.png"))
plt.close()

# in HPC with Human
# True vs Predicted --> 9899 x 2
# Best test score: 94.27%
# accuracy with best parameters: 94.39%