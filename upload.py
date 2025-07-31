from s3_utils import upload_files, download_files, list_files, download_single_file
import os
from pathlib import Path

'''# paths to files to upload 
# paths = os.listdir("resultados")

base_dir = "resultados"
paths = [os.path.join(base_dir, fname) for fname in os.listdir(base_dir)]
upload_files("experimento1", *paths)

# Example: Uploading resultados/gene_scores.csv to mcoboarr/xgboost/experimento1/gene_scores.csv'''

from s3_utils import upload_files
import os

base_dir = "resultados"
paths = [
    os.path.join(base_dir, fname)
    for fname in os.listdir(base_dir)
    if os.path.isfile(os.path.join(base_dir, fname))
]

upload_files("experimento1/human_final", *paths)
