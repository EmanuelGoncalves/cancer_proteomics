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
crispr_file = configs['crispr_file']
data_type = configs['data_type']
cell_lines_train_file = configs['cell_lines_train']
cell_lines_test_file = configs['cell_lines_test']

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

crispr = pd.read_csv(crispr_file)
crispr = crispr.melt(id_vars='Gene', var_name='Cell line name', value_name='logFC')

if data_type.lower() in ('protein', 'rna_common', 'mofa'):
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
crispr = crispr.sort_values(
    by=['Gene', 'Cell line name']).reset_index(drop=True)
# %% filtering
min_cell_lines = configs['min_cell_lines']
crispr_counts = crispr.groupby(['Gene']).size()
selected_genes = crispr_counts[crispr_counts > min_cell_lines].index.values

# selected_drugs = selected_drugs[:2]

crispr_selected = crispr[crispr['Gene'].isin(selected_genes)]
logger.info(f"Filtered genes with more than {min_cell_lines} cell lines.")
logger.info(f"Fitting {crispr_selected['Gene'].unique().size} models(genes).")

cell_lines_train = pd.read_csv(cell_lines_train_file, sep='\t')['Cell_line'].values
cell_lines_test = pd.read_csv(cell_lines_test_file, sep='\t')['Cell_line'].values

crispr_selected_train = crispr_selected[crispr_selected['Cell line name'].isin(cell_lines_train)]
crispr_selected_test = crispr_selected[crispr_selected['Cell line name'].isin(cell_lines_test)]

logger.info(f"{len(cell_lines_train)} Training cell lines: {sorted(cell_lines_train)}")
logger.info(f"{len(cell_lines_test)} Testing cell lines: {sorted(cell_lines_test)}")

# %% training
feature_df_list = []
score_df_list = []
params_df_list = []
for crispr_gene in tqdm(crispr_selected_train['Gene'].unique()):
    logger.info(f"Running GridSearchCV for Gene={crispr_gene}")

    target = configs['target']

    tmp_df_train = pd.merge(
        data_sample,
        crispr_selected_train[crispr_selected_train['Gene'] == crispr_gene][['Cell line name', target]],
        how='inner',
        left_on='Cell_line',
        right_on='Cell line name')

    tmp_df_test = pd.merge(
        data_sample,
        crispr_selected_test[crispr_selected_test['Gene'] == crispr_gene][['Cell line name', target]],
        how='inner',
        left_on='Cell_line',
        right_on='Cell line name')

    logger.info(f"Train set shape: {tmp_df_train.shape}")
    logger.info(f"Test set shape: {tmp_df_test.shape}")

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
    elif configs['imputer'] == 'zero':
        imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    elif configs['imputer'] == 'min':
        imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=configs['min_val'])
    else:
        logger.warning("No imputer selected!")

    # run grid search
    if configs['metric'] == 'corr':
        rcv = GridSearchCV(reg, param_grid, n_jobs=-1, cv=cv, scoring=corr_score, refit=True)
    else:
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
        score_dict = {'Gene': crispr_gene, 'val_mae': sign * rcv.best_score_, 'test_score': test_auc}
    else:
        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = mean_squared_error(y_test, y_pred, squared=False)
        test_r2 = r2_score(y_test, y_pred)
        test_corr = pearsonr(y_test, y_pred)[0]
        score_dict = {'Gene': crispr_gene, 'val_score': sign * rcv.best_score_,
                      'test_mae': test_mae, 'test_rmse': test_rmse,
                      'test_r2': test_r2, 'test_corr': test_corr}

    for i in range(configs['cv']):
        score_dict[f"cv{i}_{configs['metric']}"] = sign * rcv.cv_results_[f'split{i}_test_score'][rcv.best_index_]
    score_df_list.append(score_dict)

    # record best params
    params = rcv.best_params_
    logger.info(f"best param for {crispr_gene}: {params}")
    params['Gene'] = crispr_gene
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
            pd.DataFrame({configs['data_type']: X_train.columns, f'importance_{crispr_gene}': importances}))

    logger.info(f"Finished Gene={crispr_gene}\t{configs['sign'] * rcv.best_score_}\t{rcv.best_params_}")

# %% output
if 'save_scores' not in configs or configs['save_scores']:
    score_df = pd.DataFrame(score_df_list)
    score_df.to_csv(f"{configs['work_dir']}/scores_{STAMP}{log_suffix}.csv", index=False)
    params_df = pd.DataFrame(params_df_list)
    params_df.to_csv(f"{configs['work_dir']}/best_params_{STAMP}{log_suffix}.csv", index=False)

if configs["importance"]:
    feature_df = reduce(lambda x, y: pd.merge(x, y, on=configs['data_type'], how='outer'), feature_df_list)
    feature_df.to_csv(f"{configs['work_dir']}/feature_importance_{STAMP}{log_suffix}.csv", index=False)

logger.info(f"All finished.")
