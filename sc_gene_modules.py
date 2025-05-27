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
output_dir = "figures/single_gene_modules"
os.makedirs(output_dir, exist_ok=True)

# Set Scanpy to save plots 
sc.settings.figdir = output_dir

# Define file paths explicitly
expr_matrix_file = "module_expression_matrix.csv"
# meta_file = "datasets/Human/meta.tsv"

# Load module expression matrix, treating the first column (index) correctly
expr_matrix = pd.read_csv(expr_matrix_file, sep=",", header=0, index_col=0)  # index_col=0 fixes indexing issue
# Expression matrix is gene modules x cells

# Load metadata
# metadata = pd.read_csv(meta_file, sep="\t", index_col=0)
# Metadata is cells x info (cluster, age, age_unit, etc)

# Ensure cell IDs match between metadata and expression matrix
# assert metadata.index.equals(expr_matrix.columns), "Cell IDs must match between metadata and expression matrix."

# Create Anndata object
# Transpose so cells are rows and genes are columns 
# adata = ad.AnnData(X=expr_matrix.T)
# adata.obs = metadata

adata_original = sc.read_h5ad("datasets/Human/adata.h5ad")

# Transpose to cells x modules
expr_matrix = expr_matrix.T
expr_matrix.index.name = "cellId"

# Expression matrix 
# mat = adata.to_df().T  # genes as rows, cells as columns
# Metadata 
meta = adata_original.obs.copy()

# Align metadata with expression matrix
# Ensure only cells present in both are kept
shared_cells = expr_matrix.index.intersection(meta.index)
expr_matrix = expr_matrix.loc[shared_cells]
metadata = meta.loc[shared_cells]

# Create new AnnData object with module-level expression
adata = ad.AnnData(X=expr_matrix.values)
adata.obs = metadata
adata.var_names = expr_matrix.columns.astype(str)
adata.obs_names = expr_matrix.index

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
sc.pp.scale(adata)  # Standardization for PCA and clustering 

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
    'max_depth': [3, 5],
    'n_estimators': [50, 100],
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
    result_df.to_csv('true_predicted_modules.csv', index=False)
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

for class_idx in range(num_classes):  # Loop through all classes
    shap_plot_path = os.path.join(output_dir, f"shap_beeswarm_class_{class_idx}.png")

    # Create and save SHAP plot
    shap.plots.beeswarm(explanation[:, :, class_idx], show=False)  # Generate plot
    plt.savefig(shap_plot_path, bbox_inches='tight')  # Save figure
    plt.close()  # Close plot to free memory

# Compute neighbors and UMAP on the full dataset
sc.pp.pca(adata)
sc.pp.neighbors(adata, n_pcs=30)
sc.tl.umap(adata)

# Plot UMAP colored by organism
sc.pl.umap(adata, color='organism', save='_organism.png')

# Plot UMAP colored by subclass label
sc.pl.umap(adata, color='BICCN_subclass_label', save='_subclass.png')

# Compute confusion matrix
# Inverse transform to get class names back from label encoding
true_labels = label_encoder.inverse_transform(y_test)
pred_labels = label_encoder.inverse_transform(y_pred)

all_labels = label_encoder.classes_

cm = confusion_matrix(true_labels, pred_labels, labels=all_labels, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format=".2f")
plt.title("Confusion Matrix: Human (True) vs Mouse-Trained Predictions")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix_mouse_to_human.png"))
plt.close()

# in HPC with Human
# True vs Predicted: 9899 x 2
# Best test score: 71.72%
# Accuracy 71.54%