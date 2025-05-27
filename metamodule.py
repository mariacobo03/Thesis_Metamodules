# this python script creates a dataset where the columns are the 
# cells and to which metamodule they belong to 

# pandas and numpy to analyze gene expression matrices 
# instead of Seurat 
import pandas as pd # efficient for large datasets 
import numpy as np
import matplotlib.pyplot as plt # data visualization
# used for weighted gene co-expression network anlysis 
from PyWGCNA import WGCNA 
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import scanpy as sc 

#########
# SET UP
#########

# Load expression matrix
# (23460, 4026)
# mat = pd.read_csv(f"{dataset_directory}/exprMatrix.tsv.gz", sep="\t", index_col=0)

# Load the metadata
# (4026, 18)
# meta = pd.read_csv(f"{dataset_directory}/meta.tsv", sep="\t", index_col=0)

# Process expression matrix
# 23460
# genes = mat.index.str.split("|").str[-1]
# mat.index = genes

# Set argument values directly
# filepath_to_lis = "meta.tsv"
# dataset_directory = "Human"
# file_prefix = "metaatlas"
# gene_score_threshold = 0.90

# Set argument values directly
adata_file = "adata_excitory_inhibitory.h5ad"  # Path to .h5ad file
file_prefix = "metaatlas"
gene_score_threshold = 0.90  # Gene score percentile threshold

# Load AnnData object
adata = sc.read_h5ad(adata_file)

# Convert expression matrix to DataFrame (genes x cells)
mat = adata.to_df().T  # Transpose so genes are rows, matching old structure

# Simplify gene names if needed
mat.index = mat.index.str.split("|").str[-1]

# Load metadata from adata.obs
meta = adata.obs.copy()

# Ensure the metadata index matches the expression matrix columns
if not np.all(meta.index == mat.columns):
    raise ValueError("Mismatch between metadata index and expression matrix columns.")

#########
# GENE SCORE HANDLING
#########

# Load the gene scores CSV
gene_scores_file = "experimento1/gene_scores_all.csv"
# (187680, 4)
dataset_genescores = pd.read_csv(gene_scores_file)

# Replace infinite gene scores with 5000
dataset_genescores['GeneScore'].replace([np.inf, -np.inf], 5000, inplace=True)

# Add a column for the dataset
# dataset_genescores['Dataset'] = "Human"

dataset_genescores = dataset_genescores[dataset_genescores['Organism'] == 'Homo sapiens']

# Rename 'Organism' to 'Dataset' to maintain compatibility with the rest of the code
dataset_genescores.rename(columns={'Organism': 'Dataset'}, inplace=True)

metaatlas_genescores = dataset_genescores

# Save the aggregated gene scores as CSV
output_file = f"{file_prefix}_genescores.csv"
metaatlas_genescores.to_csv(output_file, index=False)
print(f"Metaatlas gene scores saved to {output_file}")


#########
# FILTRATION OF CLUSTER MARKERS BASED ON GENE SCORE
#########

# Ensure the dataset is loaded
if 'metaatlas_genescores' not in globals():
    raise ValueError("Error: The 'metaatlas_genescores' object does not exist.")

# Display basic info about the dataset
print("Clusters per dataset:")
print(dataset_genescores.groupby('Dataset')['Cluster'].nunique())
print("")
print("Cluster markers per dataset:")
print(dataset_genescores.groupby('Dataset').size())

# Distribution of gene scores
plt.figure(figsize=(10, 6))
plt.hist(dataset_genescores['GeneScore'], bins=30, color='skyblue', edgecolor='black')
plt.axvline(dataset_genescores['GeneScore'].median(), color='blue', linestyle='dashed', linewidth=1)
plt.title("Distribution of Gene Scores in metaatlas")
plt.xlabel("GeneScore")
plt.ylabel("Frequency")
# plt.show()

# Filtering by percentile threshold
threshold = dataset_genescores['GeneScore'].quantile(gene_score_threshold)
# [11965 rows x 4 columns]
filtered_genescores = dataset_genescores[dataset_genescores['GeneScore'] >= threshold]

# Effects of filtering
# Post-filtering distribution
plt.figure(figsize=(10, 6))
plt.hist(filtered_genescores['GeneScore'], bins=30, color='salmon', edgecolor='black')
plt.axvline(filtered_genescores['GeneScore'].median(), color='blue', linestyle='dashed', linewidth=1)
plt.title("Distribution of Gene Scores in metaatlas post-filter")
plt.xlabel("GeneScore")
plt.ylabel("Frequency")
# plt.show()

# Save filtered gene scores as CSV
filtered_genescores.to_csv(f"{file_prefix}_filtered_genescores.csv", index=False)
print(f"Filtered metaatlas gene scores saved to {file_prefix}_filtered_genescores.csv")

#########
# HIERARCHICAL CLUSTERING OF CLUSTER MARKERS TO GENERATE METAMODULES
#########

# Trim the filtered aggregate gene score table to contain: Gene, Cluster, GeneScore, Dataset
trimmed_filtered_genescores = filtered_genescores[['Gene', 'Cluster', 'GeneScore']]

# Pivot data to have clusters as columns and gene scores as values
pivoted_data = trimmed_filtered_genescores.pivot(index='Gene', columns='Cluster', values='GeneScore')

# Fill NA with 0
pivoted_data.fillna(0, inplace=True)

# Convert to numpy matrix for WGCNA analysis
genescores_matrix = pivoted_data.values

# Calculate Pearson correlation distance (returns a condensed matrix)
# 1D array of pairwise distances between genes 
# pdist withh produce a condensed distance matrix 

corr_matrix = np.corrcoef(genescores_matrix, rowvar=True)

# dist_matrix = pdist(genescores_matrix, metric='correlation')
# dist_matrix_square = squareform(dist_matrix)

dist_matrix_square = 1 - corr_matrix
dist_matrix_square[dist_matrix_square < 0.0000001] = 0 
dist_matrix_square = (dist_matrix_square + dist_matrix_square.T) / 2
dist_matrix_square[dist_matrix_square < 0] = 0

dist_matrix_condensed = squareform(dist_matrix_square)

# Perform hierarchical clustering using average linkage
# linkage --> merging two most similar clusters into a new one, merge all into one large cluster 
# works with condensed distance matrix

# breakpoint() 

# dist_matrix_df = pd.DataFrame(dist_matrix_square)

# Convert index to integer-based indexing
# dist_matrix_df.index = range(dist_matrix_df.shape[0])
# dist_matrix_df.columns = range(dist_matrix_df.shape[1])

# hierarchical clustering encoded as a linkage matrix 
hclust = WGCNA.hclust(dist_matrix_condensed, method='average')

# After you create dist_matrix_square_df:
dist_matrix_square_df = pd.DataFrame(
    dist_matrix_square,
    index=pd.RangeIndex(dist_matrix_square.shape[0]),  # reset index to integer range
    columns=pd.RangeIndex(dist_matrix_square.shape[1])
)

# Make sure index and columns are int type:
dist_matrix_square_df.index = dist_matrix_square_df.index.astype(int)
dist_matrix_square_df.columns = dist_matrix_square_df.columns.astype(int)

# Make sure hclust columns 0,1,3 are int:
hclust[:, [0,1,3]] = hclust[:, [0,1,3]].astype(int)

# Now pass to cutreeHybrid
metamodules = WGCNA.cutreeHybrid(
    hclust,
    cutHeight=None,
    minClusterSize=10,
    pamStage=True,
    distM=dist_matrix_square_df,
    deepSplit=1
)

# Extract cluster labels
metamodules_df = pd.DataFrame({'Gene': pivoted_data.index, 'Metamodule': metamodules['labels']})

# Save the results as a CSV
metamodules_df.to_csv("metamodules_all.csv", index=False)

print(f"Metamodules saved to _metamodules.csv")


#(Pdb) pivoted_data.shape
#(9284, 15)
#(Pdb) genescores_matrix.shape
#(9284, 15)
#(Pdb) dist_matrix.shape
#(43091686,)
#(Pdb) dist_matrix_square.shape
#(9284, 9284)
#(Pdb) dist_matrix_df.shape
#(9284, 9284)
#(Pdb) np.allclose(dist_matrix_df.values, dist_matrix_df.values.T)
#True
#(Pdb) type(dist_matrix_df)
#<class 'pandas.core.frame.DataFrame'>
#(Pdb) hclust = WGCNA.hclust(dist_matrix_df, method='average')
#(Pdb) hclust.shape
#(9283, 4)


