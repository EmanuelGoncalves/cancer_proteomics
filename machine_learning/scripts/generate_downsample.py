"""
Script of generating protein sets for down sampling analysis
"""
import pandas as pd
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path('/home/scai/SangerDrug')
protein_ruv = pd.read_csv(PROJECT_ROOT.joinpath("data/protein/E0022_P06_Protein_Matrix_ProNorM_no_control_update.txt"),
                          sep='\t', index_col=0)
proteins = protein_ruv.columns.values

# for num_proteins in list(range(500, 3500, 500)) + [3400]:
for num_proteins in [100]:
    df = {}
    for i in range(10):
        np.random.shuffle(proteins)
        df[f'run_{i}'] = np.copy(proteins[:num_proteins])
    df = pd.DataFrame(df)
    df.to_csv(PROJECT_ROOT.joinpath(f"data/protein/downsample/downsample_{num_proteins}.csv"), index=False)
