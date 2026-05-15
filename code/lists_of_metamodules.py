import pandas as pd

# Load the metamodules file
metamodules_df = pd.read_csv("experimento1/metamodules_mus_musculus.csv")

# Sort by Metamodule (optional, for consistent ordering)
metamodules_df.sort_values(by=["metamodule", "gene"], inplace=True)

# Group genes by metamodule
grouped_genes = metamodules_df.groupby('metamodule')['gene'].apply(list)

# Convert to list of lists
list_of_gene_lists = grouped_genes.tolist()

# Print the result (or return it from a function if needed)
for idx, gene_list in enumerate(list_of_gene_lists, start=1):
    print(f"Metamodule {idx}: {gene_list[:5]} ... ({len(gene_list)} genes)")  # show only first 5 genes for brevity

 

