"""
python grid_search_all.py configs/protein/mlp_grid_search.json
"""
import pandas as pd
import numpy as np
import pickle
from functools import reduce
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, make_scorer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import ElasticNet
import logging
import sys
import json
import os
from datetime import datetime
from tqdm import tqdm
from scipy.stats import pearsonr
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


def grid_search_corr(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


corr_score = make_scorer(grid_search_corr, greater_is_better=True)

STAMP = datetime.today().strftime('%Y%m%d%H%M')

config_file = sys.argv[1]

# load model configs
configs = json.load(open(config_file, 'r'))

log_suffix = ''
if 'suffix' in configs:
    log_suffix = configs['suffix']
elif len(sys.argv) > 2:
    log_suffix = sys.argv[2]

if not os.path.isdir(configs['work_dir']):
    os.system(f"mkdir -p {configs['work_dir']}")

meta_file = configs['meta_file']
data_file = configs['data_file']
cell_lines_train_file = configs['cell_lines_train']
cell_lines_test_file = configs['cell_lines_test']
ic50_file = configs['ic50_file']
data_type = configs['data_type']

log_file = f"{STAMP}{log_suffix}.log"
logger = logging.getLogger('grid_search_all')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(configs['work_dir'], log_file))
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# logger.addHandler(ch)
logger.addHandler(fh)

logger.info(open(config_file, 'r').read())
print(open(config_file, 'r').read())

seed = configs['seed']
cv = KFold(n_splits=configs['cv'], shuffle=True, random_state=seed)

logger.info(f"Running GridSearchCV for {configs['model']} with {configs['cv']}-fold CV and seed {seed}")

# %%
if configs['task'].lower() == 'classification':
    model_dict = {'rf': RandomForestClassifier,
                  'svm': SVC,
                  'en': ElasticNet,
                  'mlp': MLPClassifier}

else:
    model_dict = {'rf': RandomForestRegressor,
                  'svm': SVR,
                  'en': ElasticNet,
                  'mlp': MLPRegressor}

# %% data loading
meta = pd.read_csv(meta_file, sep='\t')

ic50 = pd.read_csv(ic50_file)

if data_type.lower() in ('protein', 'rna_common'):
    data_sample = pd.read_csv(data_file, sep='\t')
elif data_type.lower() == 'rna':
    data_raw = pd.read_csv(data_file, index_col=0).T
    data_raw.index.name = 'SIDM'
    data_raw = data_raw.reset_index()
    data_sample = pd.merge(data_raw, meta[['SIDM', 'Cell_line']].drop_duplicates()).drop(['SIDM'],
                                                                                         axis=1)
elif data_type.lower() == 'peptide':
    data_raw = pd.read_csv(data_file, sep='\t')
    data_raw_merge = pd.merge(data_raw, meta[['Automatic_MS_filename', 'Cell_line']])
    data_sample = data_raw_merge.drop(['Automatic_MS_filename'],
                                      axis=1).groupby(['Cell_line']).agg(np.nanmean).reset_index()
elif data_type.lower() in ('cancer_type', 'tissue_type'):
    data_sample = pd.read_csv(data_file)
elif data_type.lower() in ('ccle_common', 'sanger_common'):
    data_sample = pd.read_csv(data_file, sep='\t')
else:
    logger.error("Protein or RNA? Data type not supported.")
    raise Exception('data_type not supported')

logger.info(f"data matrix shape: {data_sample.shape}")

# %% shuffle so that we randomly pick data version
# ic50 = ic50.sort_values(by=['Dataset version'])
# ic50 = ic50.drop_duplicates(
#     ['Drug Id', 'Cell line name'], keep='last').sort_values(
#     by=['Drug Id', 'Cell line name']).reset_index(drop=True)
ic50 = ic50.sort_values(
    by=['Drug Id', 'Cell line name']).reset_index(drop=True)
# %% filtering
min_cell_lines = configs['min_cell_lines']
ic50_counts = ic50.groupby(['Drug Id']).size()
selected_drugs = ic50_counts[ic50_counts > min_cell_lines].index.values

# selected_drugs = selected_drugs[:2]

ic50_selected = ic50[ic50['Drug Id'].isin(selected_drugs)]
logger.info(f"Filtered drugs with more than {min_cell_lines} cell lines.")
logger.info(f"Fitting {ic50_selected['Drug Id'].unique().size} models(drugs).")

cell_lines_train = pd.read_csv(cell_lines_train_file, sep='\t')['Cell_line'].values
cell_lines_test = pd.read_csv(cell_lines_test_file, sep='\t')['Cell_line'].values

ic50_selected_train = ic50_selected[ic50_selected['Cell line name'].isin(cell_lines_train)]
ic50_selected_test = ic50_selected[ic50_selected['Cell line name'].isin(cell_lines_test)]

logger.info(f"{len(cell_lines_train)} Training cell lines: {sorted(cell_lines_train)}")
logger.info(f"{len(cell_lines_test)} Testing cell lines: {sorted(cell_lines_test)}")

# drug_list = ic50_selected_train['Drug Id'].unique()
drug_list = [1004]
for drug_id in tqdm(drug_list):
    logger.info(f"Running SFS for {drug_id}")
    target = 'rel_sensitive' if configs['task'].lower() == 'classification' else 'IC50'

    tmp_df_train = pd.merge(
        data_sample,
        ic50_selected_train[ic50_selected_train['Drug Id'] == drug_id][['Cell line name', target]],
        how='inner',
        left_on='Cell_line',
        right_on='Cell line name')
    params = configs['params']
    clf = model_dict[configs['model']](**params)
    scoring = "r2" if configs['metric'] != 'corr' else corr_score
    sfs = SFS(clf,
              k_features=tuple(configs['k_features']),
              forward=True,
              floating=False,
              verbose=2,
              scoring=scoring,
              cv=configs['cv'],
              n_jobs=-1)
    X_train = tmp_df_train.drop(['Cell_line', 'Cell line name', target], axis=1)
    y_train = tmp_df_train[target]
    X_train = X_train.fillna(-2.242616)

    sfs = sfs.fit(X_train, y_train)

    pickle.dump(sfs, open(os.path.join(configs['work_dir'], f"sfs_{drug_id}_{log_file.replace('.log', '.pkl')}"), "wb"))

    df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
    logger.info(f"\n {df.head()}")
    logger.info(f"Best score: {sfs.k_score_}")
