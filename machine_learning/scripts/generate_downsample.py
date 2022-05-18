"""
Script of generating protein sets for down sampling analysis
"""
import pandas as pd
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path('/home/scai/SangerDrug')
protein_ruv = pd.read_csv(PROJECT_ROOT.joinpath("data/protein/e0022_diann_051021_frozen_matrix_averaged_processed.txt"),
                          sep='\t', index_col=0)


protein_ruv_cellline_count = pd.DataFrame(
    protein_ruv.shape[0] - protein_ruv.isnull().sum(axis=0).sort_values(),
    columns=['Number of cell lines']).reset_index()

protein_ruv_cellline_count['percentage'] = protein_ruv_cellline_count[
    'Number of cell lines'] / protein_ruv.shape[0]

housekeeping = protein_ruv_cellline_count.query('percentage > 0.9')['index'].values
tissue_common = protein_ruv_cellline_count.query('0.2 <= percentage <= 0.9')['index'].values
tissue_rare = protein_ruv_cellline_count.query('percentage < 0.2')['index'].values

# proteins = protein_ruv.columns.values
# levels = [100] + list(range(500, 5000, 500)) + [4900]



# out_dir = Path(f"{PROJECT_ROOT}/data/protein/downsample")
# proteins = protein_ruv.columns.values

out_dir = Path(f"{PROJECT_ROOT}/data/protein/downsample_housekeeping")
proteins = housekeeping
# 2944

# out_dir = Path(f"{PROJECT_ROOT}/data/protein/downsample_tissue_common")
# proteins = tissue_common
# 3939

# out_dir = Path(f"{PROJECT_ROOT}/data/protein/downsample_tissue_rare")
# proteins = tissue_rare
# 1615

out_dir.mkdir(parents=True, exist_ok=True)
# for num_proteins in [100]:

for num_proteins in range(250, len(proteins), 250):
    df = {}
    for i in range(10):
        np.random.shuffle(proteins)
        df[f'run_{i}'] = np.copy(proteins[:num_proteins])
        tmp = protein_ruv[proteins[:num_proteins]]
        print(f"missing ratio {tmp.isna().sum().sum() / (tmp.shape[0] * tmp.shape[1])}")
    df = pd.DataFrame(df)
    df.to_csv(PROJECT_ROOT.joinpath(f"{out_dir}/downsample_{num_proteins}.csv"), index=False)
