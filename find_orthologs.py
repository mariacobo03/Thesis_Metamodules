import pandas as pd
import scanpy as sc

# Load ortholog file
ortholog_df = pd.read_csv("human_mouse_ensembl.txt")

# Create mapping dictionary: human gene name to mouse Ensembl ID
human_to_mouse = dict(zip(ortholog_df["Gene name"], ortholog_df["Mouse gene stable ID"]))

# Load datasets
human = sc.read_h5ad("human_data.h5ad")
mouse = sc.read_h5ad("mouse_data.h5ad")

# Determine which orthologs are present in both datasets
valid_orthologs = {
    h: m for h, m in human_to_mouse.items()
    if h in human.var_names and m in mouse.var_names
}

print(f"Found {len(valid_orthologs)} orthologs shared in both datasets.")
# Found 16866 orthologs shared in both datasets.

# Human
human_common = human[:, list(valid_orthologs.keys())].copy()
# Mouse
unique_mouse_genes = list(set(valid_orthologs.values()))
mouse_common = mouse[:, unique_mouse_genes].copy()