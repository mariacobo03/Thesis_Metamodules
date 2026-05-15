'''
# python script to calculate the gene scores
import pandas as pd
import numpy as np

# Load data
# adata.X = mat
mat = pd.read_csv("datasets/Human/exprMatrix.tsv.gz", sep='\t', compression='gzip')
# adata.obs = meta
meta = pd.read_csv("datasets/Human/meta.tsv", sep='\t')

# Set gene names as row names for the expression matrix

# adata.var --> columns genes OR adata.var_names

genes = mat.iloc[:, 0]  # First column contains gene names
genes = genes.str.split('|', expand=True).iloc[:, -1]  # Simplify gene names if necessary
mat = mat.drop(mat.columns[0], axis=1)  # Remove gene names from the data
mat.index = genes

# Ensure meta file has a 'cluster' column
if 'cluster' not in meta.columns:
    raise ValueError("The meta file must contain a 'cluster' column.")
'''

import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad

# Load AnnData object
# adata = sc.read_h5ad("datasets/Human/adata.h5ad")
# cells x genes

# 29486 x 14736
adata = sc.read_h5ad("adata_homo_sapiens.h5ad")
# 24213 x 14763
# adata2 = sc.read_h5ad("datasets/All/adata_excitory.h5ad")

# 53699 x 14705
'''adata = ad.concat(
    [adata1, adata2],
    join="inner",
    label="batch",
    keys=["adata_inhibitory.h5ad", "adata_excitory.h5ad"]
)'''

# Ensure 'cellId' is present as a column
adata.obs['cellId'] = adata.obs_names

# Check if required fields exist
if 'BICCN_subclass_label' not in adata.obs.columns:
    raise ValueError("The .h5ad file must contain a 'subclass_label' column in adata.obs.")

if 'cellId' not in adata.obs.columns:
    raise ValueError("The .h5ad file must contain a 'cellId' column in adata.obs.")

# Extract expression matrix (dense if necessary)
mat = adata.to_df().T  # genes as rows, cells as columns

# Gene names
genes = mat.index

# Metadata
meta = adata.obs[['cellId', 'BICCN_subclass_label']].copy()

# Initialize list to store gene scores
gene_scores = []

# Calculate gene scores for each cluster
clusters = meta['BICCN_subclass_label'].unique()
for cluster_id in clusters:
    # Get cells belonging to the current cluster
    cluster_cells = meta.loc[meta['BICCN_subclass_label'] == cluster_id, 'cellId']
    
    # Check that these cells exist in the expression matrix
    if not all(cell in mat.columns for cell in cluster_cells):
        raise ValueError(f"Some cluster cells for cluster {cluster_id} are not found in the expression matrix.")
    
    # Subset expression matrix for the cluster and non-cluster cells
    cluster_expr = mat[cluster_cells]
    non_cluster_cells = mat.drop(columns=cluster_cells)
    non_cluster_expr = non_cluster_cells

    # Calculate percentages of expressing cells
    pct_in_cluster = (cluster_expr > 0).mean(axis=1)
    pct_not_in_cluster = (non_cluster_expr > 0).mean(axis=1)
    
    # Compute average expression for log2-fold-change
    avg_expr_in_cluster = cluster_expr.mean(axis=1)
    avg_expr_not_in_cluster = non_cluster_expr.mean(axis=1)
    
    # Avoid division by zero
    pct_not_in_cluster = pct_not_in_cluster.replace(0, 1e-10)
    avg_expr_not_in_cluster = avg_expr_not_in_cluster.replace(0, 1e-10)

    # Find dominant organism for this cluster
    organism_counts = adata.obs.loc[cluster_cells, 'organism'].value_counts()

    # Calculate gene scores
    gene_scores.append(pd.DataFrame({
        'Gene': mat.index,
        'Cluster': cluster_id,
        'GeneScore': (pct_in_cluster / pct_not_in_cluster) * 
                     np.log2(avg_expr_in_cluster / avg_expr_not_in_cluster)
    }))

# 220575 x 4 (Gene, Organism, Cluster, GeneScore)
# Combine gene scores into a single DataFrame
gene_scores_df = pd.concat(gene_scores, ignore_index=True)

# Save results
gene_scores_df.to_csv("gene_scores.csv", index=False)
