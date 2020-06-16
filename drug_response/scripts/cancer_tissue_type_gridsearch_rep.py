import pandas as pd
import numpy as np
from functools import reduce
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import ElasticNet
import logging
import sys
import json
import os
from datetime import datetime
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
import pickle

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

data_file = configs['data_file']
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
logger.addHandler(ch)
logger.addHandler(fh)

logger.info(open(config_file, 'r').read())

seed = configs['seed']
cv = GroupShuffleSplit(n_splits=configs['cv'], random_state=seed)

logger.info(f"Running GridSearchCV for {configs['model']} with {configs['cv']}-fold CV and seed {seed}")

# %%
model_dict = {'rf': RandomForestClassifier(n_jobs=-1),
              'svm': SVC(),
              'en': ElasticNet(),
              'svm-linear': SVC(kernel='linear'),
              'mlp': MLPClassifier(),
              'xgboost': XGBClassifier(n_jobs=-1),
              'lgbm': LGBMClassifier(n_jobs=-1)}

data_sample = pd.read_csv(configs['data_file'], sep='\t')
logger.info(f"data matrix shape: {data_sample.shape}")

type_count = data_sample.groupby([configs['target']]).size()
selected_types = type_count[type_count > configs['cut_off']].index.values
data_sample = data_sample[data_sample[configs['target']].isin(selected_types)].reset_index(drop=True)
proteins = [x for x in data_sample.columns if '_HUMAN' in x]

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

cell_lines = data_sample[['Cell_line', configs['target']]].drop_duplicates().reset_index(drop=True)
train_cell_lines, test_cell_lines = train_test_split(cell_lines['Cell_line'], random_state=seed, shuffle=True,
                                                     test_size=0.2,
                                                     stratify=cell_lines[configs['target']])
X_train = data_sample[data_sample['Cell_line'].isin(train_cell_lines)][proteins]
X_train_groups = data_sample[data_sample['Cell_line'].isin(train_cell_lines)]['Cell_line']
X_test = data_sample[data_sample['Cell_line'].isin(test_cell_lines)][proteins]
y_train = data_sample[data_sample['Cell_line'].isin(train_cell_lines)][configs['target']]
y_test = data_sample[data_sample['Cell_line'].isin(test_cell_lines)][configs['target']]

if imputer:
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

logger.info(X_train.shape)
logger.info(X_test.shape)
clf = model_dict[configs['model']]
param_grid = configs['params_grid']
rcv = GridSearchCV(clf, param_grid, n_jobs=1, cv=cv, scoring=configs['metric'], refit=True, verbose=3)
rcv.fit(X_train, y_train, groups=X_train_groups)
y_pred = rcv.best_estimator_.predict(X_test)

feature_df_list = []
score_df_list = []
params_df_list = []
test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average='macro')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
score_dict = {'val_score': rcv.best_score_, 'test_accuracy': test_accuracy, 'test_f1': test_f1}
for i in range(configs['cv']):
    score_dict[f"cv{i}_{configs['metric']}"] = rcv.cv_results_[f'split{i}_test_score'][rcv.best_index_]
score_df_list.append(score_dict)

params = rcv.best_params_
print(params)
params_df_list.append(params)

# record feature importance if possible
importances = None
if configs["importance"]:
    if configs['model'] in ('rf', 'lgbm', 'xgboost'):
        importances = rcv.best_estimator_.feature_importances_
        explainer = shap.TreeExplainer(rcv.best_estimator_)
        # shap_values = explainer.shap_values(X_train)
    elif configs['model'] in ('svm-linear'):
        importances = rcv.best_estimator_.coef_[0]
    elif configs['model'] in ('en'):
        importances = rcv.best_estimator_.coef_
    else:
        logger.error("feature importance is not supported for the current model.")

    feature_df_list.append(
        pd.DataFrame({configs['data_type']: proteins, f'importance': importances}))

if 'save_scores' not in configs or configs['save_scores']:
    score_df = pd.DataFrame(score_df_list)
    score_df.to_csv(f"{configs['work_dir']}/scores_{STAMP}{log_suffix}.csv", index=False)
    params_df = pd.DataFrame(params_df_list)
    params_df.to_csv(f"{configs['work_dir']}/best_params_{STAMP}{log_suffix}.csv", index=False)
    logger.info(score_df)

if configs["importance"]:
    feature_df = reduce(lambda x, y: pd.merge(x, y, on=configs['data_type'], how='outer'), feature_df_list)
    feature_df.to_csv(f"{configs['work_dir']}/feature_importance_{STAMP}{log_suffix}.csv", index=False)
    # if configs['model'] in ('rf', 'lgbm', 'xgboost'):
    #     pickle.dump(shap_values, open(f"{configs['work_dir']}/shap_{STAMP}{log_suffix}.pkl", "wb"))

logger.info(f"All finished.")
