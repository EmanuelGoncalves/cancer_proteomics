"""
python grid_search_all.py configs/protein/mlp_grid_search.json
"""
import pandas as pd
import numpy as np
import pickle
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score
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

STAMP = datetime.today().strftime('%Y%m%d%H%M')

config_file = sys.argv[1]

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

logger.info(f"Running GridSearchCV for {configs['model']} with {configs['cv']}-fold CV and seed {seed}")

# %%
if configs['task'].lower() == 'classification':
    model_dict = {'rf': RandomForestClassifier(),
                  'svm': SVC(),
                  'en': ElasticNet(),
                  'svm-linear': SVC(kernel='linear'),
                  'mlp': MLPClassifier()}

else:
    model_dict = {'rf': RandomForestRegressor(),
                  'svm': SVR(),
                  'en': ElasticNet(),
                  'svm-linear': SVR(kernel='linear'),
                  'mlp': MLPRegressor()}

# %% data loading
meta = pd.read_csv(meta_file, sep='\t')

ic50 = pd.read_csv(ic50_file)

if data_type.lower() == 'protein':
    data_raw = pd.read_csv(data_file, sep='\t')
    data_raw = data_raw.rename(columns={'Unnamed: 0': 'Automatic_MS_filename'})
    data_raw_merge = pd.merge(data_raw, meta[['Automatic_MS_filename', 'Cell_line']])

    data_sample = data_raw_merge.drop(['Automatic_MS_filename'],
                                      axis=1).groupby(['Cell_line']).agg(np.nanmean).reset_index()
elif data_type.lower() == 'rna':
    data_raw = pd.read_csv(data_file, index_col=0).T
    data_raw.index.name = 'SIDM'
    data_raw = data_raw.reset_index()
    data_sample = pd.merge(data_raw, meta[['SIDM', 'Cell_line']].drop_duplicates()).drop(['SIDM'],
                                                                                         axis=1)
elif data_type.lower() == 'peptide':
    data_raw = pickle.load(open(data_file, 'rb'))
    data_raw_merge = pd.merge(data_raw, meta[['Automatic_MS_filename', 'Cell_line']])
    data_sample = data_raw_merge.drop(['Automatic_MS_filename'],
                                      axis=1).groupby(['Cell_line']).agg(np.nanmean).reset_index()
elif data_type.lower() in ('cancer_type', 'tissue_type'):
    data_sample = pd.read_csv(data_file)
    data_raw = data_sample
elif data_type.lower() in ('ccle_common', 'sanger_common'):
    data_sample = pd.read_csv(data_file, sep='\t')
    data_raw = data_sample
else:
    logger.error("Protein or RNA? Data type not supported.")
    raise Exception('data_type not supported')

logger.info(f"data matrix shape: {data_sample.shape}")

# %% shuffle so that we randomly pick data version
ic50 = ic50.sort_values(by=['Dataset version'])
ic50 = ic50.drop_duplicates(
    ['Drug Id', 'Cell line name'], keep='last').sort_values(
    by=['Drug Id', 'Cell line name']).reset_index(drop=True)

# %% filtering
min_cell_lines = configs['min_cell_lines']
ic50_counts = ic50.groupby(['Drug Id']).size()
selected_drugs = ic50_counts[ic50_counts > min_cell_lines].index.values

# selected_drugs = selected_drugs[:2]

ic50_selected = ic50[ic50['Drug Id'].isin(selected_drugs)]
logger.info(f"Filtered drugs with more than {min_cell_lines} cell lines.")
logger.info(f"Fitting {ic50_selected['Drug Id'].unique().size} models(drugs).")

cell_lines_train, cell_lines_test = train_test_split(sorted(data_sample['Cell_line'].unique()),
                                                     test_size=0.2,
                                                     random_state=seed)

ic50_selected_train = ic50_selected[ic50_selected['Cell line name'].isin(cell_lines_train)]
ic50_selected_test = ic50_selected[ic50_selected['Cell line name'].isin(cell_lines_test)]

logger.info(f"{len(cell_lines_train)} Training cell lines: {sorted(cell_lines_train)}")
logger.info(f"{len(cell_lines_test)} Testing cell lines: {sorted(cell_lines_test)}")

# %% training
feature_df_list = []
score_df_list = []
params_df_list = []
for drug_id in tqdm(ic50_selected_train['Drug Id'].unique()):
    logger.info(f"Running GridSearchCV for Drug Id={drug_id}")

    target = 'rel_sensitive' if configs['task'].lower() == 'classification' else 'IC50'
    if 'target' in configs:
        target = configs['target']

    tmp_df_train = pd.merge(
        data_sample,
        ic50_selected_train[ic50_selected_train['Drug Id'] == drug_id][['Cell line name', target]],
        how='inner',
        left_on='Cell_line',
        right_on='Cell line name')

    tmp_df_test = pd.merge(
        data_sample,
        ic50_selected_test[ic50_selected_test['Drug Id'] == drug_id][['Cell line name', target]],
        how='inner',
        left_on='Cell_line',
        right_on='Cell line name')

    X_train = tmp_df_train.drop(['Cell_line', 'Cell line name', target], axis=1)
    X_test = tmp_df_test.drop(['Cell_line', 'Cell line name', target], axis=1)

    y_train = tmp_df_train[target]
    y_test = tmp_df_test[target]

    reg = model_dict[configs['model']]
    param_grid = configs['params_grid']

    # impute NA
    imputer = None
    if configs['imputer'] == 'mean':
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    elif configs['imputer'] == 'KNN':
        imputer = KNNImputer(missing_values=np.nan)
    else:
        logger.warning("No imputer selected!")

    # run grid search
    rcv = GridSearchCV(reg, param_grid, n_jobs=-1, cv=cv, scoring=configs['metric'], refit=True)
    if configs['imputer'].lower() == 'none':
        rcv.fit(X_train, y_train)
        y_pred = rcv.best_estimator_.predict(X_test)
    else:
        rcv.fit(imputer.fit_transform(X_train), y_train)
        y_pred = rcv.best_estimator_.predict(imputer.transform(X_test))

    # gather results
    if 'sign' in configs:
        sign = configs['sign']
    else:
        sign = -1 if configs['task'] == 'regression' else 1

    if configs['task'].lower() == 'classification':
        test_auc = roc_auc_score(y_test, y_pred)
        score_dict = {'Drug Id': drug_id, 'val_mae': sign * rcv.best_score_, 'test_score': test_auc}
    else:
        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = mean_squared_error(y_test, y_pred, squared=False)
        test_r2 = r2_score(y_test, y_pred)
        test_corr = pearsonr(y_test, y_pred)[0]
        score_dict = {'Drug Id': drug_id, 'val_score': sign * rcv.best_score_,
                              'test_mae': test_mae, 'test_rmse': test_rmse,
                              'test_r2': test_r2, 'test_corr': test_corr}

    for i in range(configs['cv']):
        score_dict[f"cv{i}_{configs['metric']}"] = sign * rcv.cv_results_[f'split{i}_test_score'][rcv.best_index_]
    score_df_list.append(score_dict)

    # record best params
    params = rcv.best_params_
    params['Drug Id'] = drug_id
    params_df_list.append(params)

    # record feature importance if possible
    importances = None
    if configs["importance"]:
        if configs['model'] in ('rf', 'lgbm'):
            importances = rcv.best_estimator_.feature_importances_
        elif configs['model'] in ('svm-linear'):
            importances = rcv.best_estimator_.coef_[0]
        elif configs['model'] in ('en'):
            importances = rcv.best_estimator_.coef_
        else:
            logger.error("feature importance is not supported for the current model.")

        feature_df_list.append(
            pd.DataFrame({'protein': X_train.columns, f'importance_{drug_id}': importances}))

    logger.info(f"Finished Drug Id={drug_id}\t{configs['sign'] * rcv.best_score_}\t{rcv.best_params_}")

# %% output
score_df = pd.DataFrame(score_df_list)
score_df.to_csv(f"{configs['work_dir']}/scores_{STAMP}.csv", index=False)
params_df = pd.DataFrame(params_df_list)
params_df.to_csv(f"{configs['work_dir']}/best_params_{STAMP}.csv", index=False)

if configs["importance"]:
    feature_df = reduce(lambda x, y: pd.merge(x, y, on='protein', how='outer'), feature_df_list)
    feature_df.to_csv(f"{configs['work_dir']}/feature_importance_{STAMP}.csv", index=False)
