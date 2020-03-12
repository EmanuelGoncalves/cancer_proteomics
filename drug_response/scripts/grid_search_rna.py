"""
This script is deprecated.
"""
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.svm import SVR, NuSVR
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet
import logging
import sys
import json
import os
from datetime import datetime
from tqdm import tqdm

STAMP = datetime.today().strftime('%Y%m%d%H%M')

config_file = sys.argv[1]

# load model configs
configs = json.load(open(config_file, 'r'))
if not os.path.isdir(configs['work_dir']):
    os.system(f"mkdir -p {configs['work_dir']}")

meta_file = configs['meta_file']
rna_file = configs['rna_file']
ic50_file = configs['ic50_file']

log_file = f"{STAMP}.log"
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

seed = configs['seed']
cv = KFold(n_splits=configs['cv'], shuffle=True, random_state=seed)

logger.info(f"Running GridSearchCV for {configs['model']} with {configs['cv']}-fold CV and seed {seed}")

# %%
model_dict = {'rf': RandomForestRegressor(),
              'svm': SVR(),
              'en': ElasticNet(),
              'svm-linear': SVR(kernel='linear'),
              'mlp': MLPRegressor()}

# %% data loading
meta = pd.read_csv(meta_file, sep='\t')
rna_raw = pd.read_csv(rna_file, index_col=0).T
ic50 = pd.read_csv(ic50_file)

rna_raw.index.name = 'SIDM'
rna_raw = rna_raw.reset_index()
rna_sample = pd.merge(rna_raw, meta[['SIDM', 'Cell_line']].drop_duplicates()).drop(['SIDM'],
                                                                                   axis=1)

# shuffle so that we randomly pick data version
ic50_shuffle = ic50.sample(frac=1, random_state=seed).reset_index(drop=True).drop_duplicates(
    ['Drug Id', 'Cell line name'])

# filtering
min_cell_lines = configs['min_cell_lines']
ic50_shuffle_counts = ic50_shuffle.groupby(['Drug Id']).size()
selected_drugs = ic50_shuffle_counts[ic50_shuffle_counts > min_cell_lines].index.values

# selected_drugs = selected_drugs[:2]

ic50_shuffle_selected = ic50_shuffle[ic50_shuffle['Drug Id'].isin(selected_drugs)]
logger.info(f"Filtered drugs with more than {min_cell_lines} cell lines.")
logger.info(f"Fitting {ic50_shuffle_selected['Drug Id'].unique().size} models(drugs).")

cell_lines_train, cell_lines_test = train_test_split(sorted(ic50_shuffle_selected['Cell line name'].unique()),
                                                     test_size=0.2,
                                                     random_state=seed)

ic50_shuffle_selected_train = ic50_shuffle_selected[ic50_shuffle_selected['Cell line name'].isin(cell_lines_train)]
ic50_shuffle_selected_test = ic50_shuffle_selected[ic50_shuffle_selected['Cell line name'].isin(cell_lines_test)]

logger.info(f"Training cell lines: {sorted(cell_lines_train)}")
logger.info(f"Testing cell lines: {sorted(cell_lines_test)}")

# %% training
feature_df_list = []
score_df_list = []
params_df_list = []
for drug_id in tqdm(ic50_shuffle_selected_train['Drug Id'].unique()):
    logger.info(f"Running GridSearchCV for Drug Id={drug_id}")
    tmp_df_train = pd.merge(
        rna_sample,
        ic50_shuffle_selected_train[ic50_shuffle_selected_train['Drug Id'] == drug_id][['Cell line name', 'IC50']],
        how='inner',
        left_on='Cell_line',
        right_on='Cell line name')
    X_train = tmp_df_train.drop(['Cell_line', 'Cell line name', 'IC50'], axis=1)
    y_train = tmp_df_train['IC50']

    tmp_df_test = pd.merge(
        rna_sample,
        ic50_shuffle_selected_test[ic50_shuffle_selected_test['Drug Id'] == drug_id][['Cell line name', 'IC50']],
        how='inner',
        left_on='Cell_line',
        right_on='Cell line name')
    X_test = tmp_df_test.drop(['Cell_line', 'Cell line name', 'IC50'], axis=1)
    y_test = tmp_df_test['IC50']

    reg = model_dict[configs['model']]
    param_grid = configs['params_grid']

    imputer = None
    if configs['imputer'] == 'mean':
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    elif configs['imputer'] == 'KNN':
        imputer = KNNImputer(missing_values=np.nan)
    else:
        logger.warning("No imputer selected!")

    rcv = GridSearchCV(reg, param_grid, n_jobs=-1, cv=cv, scoring='neg_mean_absolute_error', refit=True)
    rcv.fit(imputer.fit_transform(X_train), y_train)
    y_pred = rcv.best_estimator_.predict(imputer.transform(X_test))
    test_mae = mean_absolute_error(y_test, y_pred)
    score_df_list.append(
        pd.DataFrame({'Drug Id': [drug_id], 'val_mae': [-1 * rcv.best_score_], 'test_mae': [test_mae]}))
    params = rcv.best_params_
    params['Drug Id'] = drug_id
    params_df_list.append(params)

    importances = None
    if configs["importance"]:
        if configs['model'] == 'rf':
            importances = rcv.best_estimator_.feature_importances_
        elif configs['model'] in ('svm-linear'):
            importances = rcv.best_estimator_.coef_[0]
        elif configs['model'] in ('en'):
            importances = rcv.best_estimator_.coef_
        else:
            logger.error("feature importance is not supported for the current model.")

        feature_df_list.append(
            pd.DataFrame({'protein': X_train.columns, f'importance_{drug_id}': importances}))

    logger.info(f"Finished Drug Id={drug_id}\t{-1 * rcv.best_score_}\t{rcv.best_params_}")

# %% output
score_df = pd.concat(score_df_list)
score_df.to_csv(f"{configs['work_dir']}/scores_{STAMP}.csv", index=False)
params_df = pd.DataFrame(params_df_list)
params_df.to_csv(f"{configs['work_dir']}/best_params_{STAMP}.csv", index=False)

if configs["importance"]:
    feature_df = reduce(lambda x, y: pd.merge(x, y, on='protein', how='outer'), feature_df_list)
    feature_df.to_csv(f"{configs['work_dir']}/feature_importance_{STAMP}.csv", index=False)
