import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
import logging
import sys
import json
import os
from datetime import datetime
from tqdm import tqdm, trange
from scipy.stats import pearsonr
import time
from sklearn.model_selection import GridSearchCV

STAMP = datetime.today().strftime('%Y%m%d%H%M')

config_file = sys.argv[1]

# load model configs
configs = json.load(open(config_file, 'r'))

meta_file = configs['meta_file']
data_file = configs['data_file']
ic50_file = configs['ic50_file']
data_type = configs['data_type']

logger = logging.getLogger('mlp-debug')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# load model configs
configs = json.load(open(config_file, 'r'))
if not os.path.isdir(configs['work_dir']):
    os.system(f"mkdir -p {configs['work_dir']}")

meta_file = configs['meta_file']
data_file = configs['data_file']
ic50_file = configs['ic50_file']
data_type = configs['data_type']

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

meta = pd.read_csv(meta_file, sep='\t')
ic50 = pd.read_csv(ic50_file)
ic50 = ic50.sort_values(by=['Dataset version'])
ic50 = ic50.drop_duplicates(
    ['Drug Id', 'Cell line name'], keep='last').sort_values(
    by=['Drug Id', 'Cell line name']).reset_index(drop=True)
min_cell_lines = configs['min_cell_lines']
ic50_counts = ic50.groupby(['Drug Id']).size()
selected_drugs = ic50_counts[ic50_counts > min_cell_lines].index.values

ic50_selected = ic50[ic50['Drug Id'].isin(selected_drugs)]
ic50_selected_pivot = pd.pivot(ic50_selected[['Cell line name', 'Drug Id', 'IC50_norm']], index='Cell line name',
                               columns='Drug Id', values='IC50_norm').reset_index()

ic50_selected_pivot = ic50_selected_pivot.sort_values(by=['Cell line name']).reset_index(drop=True)

data_raw = pd.read_csv(data_file, sep='\t')
data_raw = data_raw.rename(columns={'Unnamed: 0': 'Automatic_MS_filename'})
data_raw_merge = pd.merge(data_raw, meta[['Automatic_MS_filename', 'Cell_line']])

data_sample = data_raw_merge.drop(['Automatic_MS_filename'],
                                  axis=1).groupby(['Cell_line']).agg(np.nanmean).reset_index()
data_sample = data_sample.sort_values(by=['Cell_line']).reset_index(drop=True)

cell_lines_train, cell_lines_test = train_test_split(sorted(ic50_selected['Cell line name'].unique()),
                                                     test_size=0.2,
                                                     random_state=seed)

logger.info(f"Training cell line: {cell_lines_train}")
logger.info(f"Test cell line: {cell_lines_test}")

ic50_selected_pivot_train = ic50_selected_pivot[
    ic50_selected_pivot['Cell line name'].isin(cell_lines_train)].reset_index(drop=True)
ic50_selected_pivot_test = ic50_selected_pivot[
    ic50_selected_pivot['Cell line name'].isin(cell_lines_test)].reset_index(drop=True)

num_of_drugs = ic50_selected_pivot_train.shape[1] - 1

data_sample_train = data_sample[data_sample['Cell_line'].isin(cell_lines_train)].reset_index(drop=True)
data_sample_test = data_sample[data_sample['Cell_line'].isin(cell_lines_test)].reset_index(drop=True)

merged_df_train = pd.merge(data_sample_train, ic50_selected_pivot_train, left_on=['Cell_line'],
                           right_on=['Cell line name']).drop(['Cell line name'], axis=1)
merged_df_test = pd.merge(data_sample_test, ic50_selected_pivot_test, left_on=['Cell_line'],
                          right_on=['Cell line name']).drop(['Cell line name'], axis=1)


train_ic50 = merged_df_train.iloc[:, -num_of_drugs:]
val_ic50 = merged_df_test.iloc[:, -num_of_drugs:]
train_df = merged_df_train.iloc[:, 1:(merged_df_train.shape[1] - num_of_drugs)]
val_df = merged_df_test.iloc[:, 1:(merged_df_train.shape[1] - num_of_drugs)]

param_grid = configs['params_grid']
imputer = None
if configs['imputer'] == 'mean':
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
elif configs['imputer'] == 'KNN':
    imputer = KNNImputer(missing_values=np.nan)
else:
    logger.warning("No imputer selected!")

model = MLPRegressor()
sign = configs['sign']
rcv = GridSearchCV(model, param_grid, n_jobs=-1, cv=cv, scoring=configs['metric'], refit=True)
rcv.fit(imputer.fit_transform(train_df), train_ic50)
y_pred = rcv.best_estimator_.predict(imputer.transform(val_df))
y_test = val_ic50

test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = mean_squared_error(y_test, y_pred, squared=False)
test_r2 = r2_score(y_test, y_pred)
test_corr = pearsonr(y_test, y_pred)[0]
score_dict = {'val_score': sign * rcv.best_score_,
              'test_mae': test_mae, 'test_rmse': test_rmse,
              'test_r2': test_r2, 'test_corr': test_corr}
for i in range(configs['cv']):
    score_dict[f"cv{i}_{configs['metric']}"] = sign * rcv.cv_results_[f'split{i}_test_score'][rcv.best_index_]
logger.info(score_dict)
