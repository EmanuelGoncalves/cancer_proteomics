import pandas as pd
import numpy as np

peptide_raw = pd.read_csv("../data/protein/E0022_P06_Peptide_Matrix_RUV.tsv", sep='\t')
peptide_raw = peptide_raw.rename(columns={'index': 'Automatic_MS_filename'})
meta = pd.read_csv('/home/scai/SangerDrug/data/E0022_P06_final_sample_map_no_control.txt', sep='\t')
peptide_merge = pd.merge(peptide_raw, meta[['Automatic_MS_filename', 'Cell_line']])
peptide_sample = peptide_merge.drop(['Automatic_MS_filename'],
                                 axis=1).groupby(['Cell_line']).agg(np.nanmean).reset_index()
peptide_sample.to_csv("../data/protein/E0022_P06_Peptide_Matrix_RUV_avg.tsv", sep='\t', index=False)