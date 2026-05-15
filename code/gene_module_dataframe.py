import numpy as np 
import pandas as pd
import scanpy as sc

# Load data 
# Gene module table --> gene x gene module
# gene_module = pd.read_csv("metaatlas_metamodules.csv")
# Covert into dictionary: keys = genes & values = gene module 
# gene_to_module = gene_module.set_index('genes')['metamodules'].to_dict()

# Expression matrix 
# expr_matrix = pd.read_csv("exprMatrix.tsv.gz",
#                           sep="\t", compression="gzip", header=0, index_col=0)
# have same gene names as gene_module 
# expr_matrix.index = expr_matrix.index.str.split('|').str[0]
# expr_matrix = expr_matrix.fillna(0)  # Replace NaNs with 0 instead of dropping

# print(expr_matrix.sum(axis=0) # sum of columns 
# print(expr_matrix.sum(axis=1) # sum of rows 

# Load expression data from .h5ad
adata = sc.read_h5ad("adata_callithrix_jacchus.h5ad")

# Convert expression matrix to dense DataFrame with gene names as index
expr_matrix = adata.to_df().T  # genes x cells

# Optionally simplify gene names (if needed, based on original script logic)
expr_matrix.index = expr_matrix.index.str.split('|').str[0]

# Replace NaNs with 0
expr_matrix = expr_matrix.fillna(0)

# Load gene module table (still from CSV)
gene_module = pd.read_csv("experimento1/metamodules_mus_musculus.csv")

# Get unique modules and create a mapping
unique_modules = gene_module['metamodules'].unique()
module_to_index = {module: i for i, module in enumerate(unique_modules)}
# {2: 0, 4: 1, 3: 2, 22: 3, 1: 4, 6: 5, 5: 6, 10: 7, 8: 8, 13: 9, 17: 10, 11: 11, 12: 12, 
# 9: 13, 7: 14, 14: 15, 23: 16, 18: 17, 21: 18, 20: 19, 15: 20, 19: 21, 16: 22, 24: 23}

# Create a binary mapping matrix (one-hot encoding of genes --> modules)
# up to 23459
# gene_list --> ['1/2-SBSRNA4', 'A1BG', 'A1BG-AS1', ...]
gene_list = expr_matrix.index.tolist()

# creating an empty matrix with dimensions of genes and unique modules 
# row represents module 
mapping_matrix = np.zeros((len(unique_modules), len(gene_list)))
# matrix shape --> (24, 23460)
# print(mapping_matrix.sum(axis=1))

# 1s in columns corresponding to genes in that module 
for i, genes in enumerate(gene_list):
    # if gene present in gene_module 
    # assign correspoding module 
    if genes in gene_module['genes'].values:
        module = gene_module.loc[gene_module['genes'] == genes, 'metamodules'].values[0]
        module_index = module_to_index[module]
        # sets 1 on the correct row and column 
        mapping_matrix[module_index, i] = 1  # One-hot encoding


# Convert expression matrix to a numpy array 
# gene expression levels per cell 
expression_values = np.array(expr_matrix, dtype=float)
# shape expression_values --> (23460, 4025) --> genes x cells 


# Multiply mapping matrix with expression matrix
# module_expression_values = mapping_matrix @ expression_values  # (Modules x Genes) x (Genes x Cells) → (Modules x Cells)
# The multiplication only happens where a 1 exists
# All genes belonging to the same module get summed together 
# Result: total expression of all genes in that module 
module_expression_values = np.dot(mapping_matrix, expression_values)
# shape --> (24, 4025)


# Normalize by number of genes per module
# Module's expression us averaged rather than summed 
gene_counts = mapping_matrix.sum(axis=1).reshape(-1, 1)  # Get count of genes per module
module_expression_values /= gene_counts  # Element-wise division

# Convert back to DataFrame
module_expression_df = pd.DataFrame(module_expression_values, index=unique_modules, columns=expr_matrix.columns)

# Reset index to include metamodule IDs as a column
module_expression_df = module_expression_df.reset_index()  # Moves index (Metamodule) into a column
module_expression_df.rename(columns={"index": "Metamodule"}, inplace=True)  # Rename it explicitly

# Save the output with the correct structure
module_expression_df.to_csv("module_expression_matrix_marmoset.csv", index=False)
