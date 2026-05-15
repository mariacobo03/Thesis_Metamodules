from s3_utils import upload_files, download_files, list_files, download_single_file
import os
from pathlib import Path


here = Path(__file__).parent.absolute()
data_path = os.path.join(here, ".", "datasets", "Human")
os.makedirs(data_path, exist_ok=True)
#I probably have to change this to my other folder s3://braingeneersdev/jgf/ryan_vik/
# download_single_file(
#     key="scRNA/rnh027_filtered_feature_bc_matrix.h5",
#     local_path=os.path.join(data_path, "rnh027_filtered_feature_bc_matrix"),
# )

# download_single_file(
#     key="scRNA/rnh029_filtered_feature_bc_matrix.h5",
#     local_path=os.path.join(data_path, "rnh029_filtered_feature_bc_matrix.h5ad"),
# )

download_single_file(key="jgf/jing_models/allen-human/human-cortex/adata.h5ad",local_path=os.path.join(data_path,"adata_human.h5ad"))

download_single_file(key="jgf/maria/results/adata_maria.h5ad",
    local_path=os.path.join(here, "adata_maria.h5ad"))

download_single_file(key="mcoboarr/xgboost/experimento1/adata_excitory_inhibitory.h5ad",
    local_path=os.path.join(here, "adata_excitory_inhibitory.h5ad"))

download_single_file(key="mcoboarr/xgboost/experimento1/adata_homo_sapiens.h5ad",
    local_path=os.path.join(here, "adata_homo_sapiens.h5ad"))

download_single_file(key="mcoboarr/xgboost/experimento1/adata_mus_musculus.h5ad",
    local_path=os.path.join(here, "adata_mus_musculus.h5ad"))

download_single_file(key="mcoboarr/xgboost/experimento1/adata_callithrix_jacchus.h5ad",
    local_path=os.path.join(here, "adata_callithrix_jacchus.h5ad"))

