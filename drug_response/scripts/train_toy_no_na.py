import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.svm import SVR, NuSVR
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

seed = 42

cv5 = KFold(n_splits=5, shuffle=True, random_state=seed)
# %%
meta = pd.read_csv("../data/E0022_P01-P05_sample_map.txt", sep='\t')
protein_raw = pd.read_csv("../data/E0022_P05_protein_intensities.txt", sep='\t')
ic50 = pd.read_csv("../data/DrugResponse_PANCANCER_GDSC1_GDSC2_IC_20191119.csv")

protein_raw = protein_raw.rename(columns={'Unnamed: 0': 'Automatic_MS_filename'})
protein_raw_merge = pd.merge(protein_raw, meta[['Automatic_MS_filename', 'Cell_line']])

protein_sample_avg = protein_raw_merge.drop(['Automatic_MS_filename'],
                                            axis=1).groupby(['Cell_line']).agg(np.nanmean).reset_index()
protein_sample_avg = protein_sample_avg.dropna(axis=1)
# shuffle so that we randomly pick data version
ic50_shuffle = ic50.sample(frac=1).reset_index(drop=True).drop_duplicates(
    ['Drug Id', 'Cell line name'])

# %%
drug_id = 1001
tmp_df = pd.merge(
    protein_sample_avg,
    ic50_shuffle[ic50_shuffle['Drug Id'] == drug_id][['Cell line name', 'IC50']],
    how='inner',
    left_on='Cell_line',
    right_on='Cell line name')

X = tmp_df.drop(['Cell_line', 'Cell line name', 'IC50'], axis=1)
y = tmp_df['IC50']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

# %% RF
reg = RandomForestRegressor(n_jobs=-1)
param_grid = {'n_estimators': [500, 1000, 2000]}

rcv = GridSearchCV(reg, param_grid, n_jobs=1, cv=cv5, scoring='r2')
rcv.fit(X_train, y_train)
print(rcv.best_score_)  # 0.22535313276929042
print(rcv.best_params_)  # {'n_estimators': 2000}

# %% LGBM without imputation
reg = LGBMRegressor()
param_grid = {'num_iterations': [300, 400, 500], 'max_bin': [150, 200, 255], 'learning_rate': [1e-1, 5e-2],
              'num_leaves': [15, 20, 31]}
rcv = GridSearchCV(reg, param_grid, n_jobs=1, cv=cv5, scoring='r2')
rcv.fit(X_train, y_train)
print(rcv.best_score_)
print(rcv.best_params_)

# %%
reg = XGBRegressor(n_estimators=100)
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
print(np.mean(cross_val_score(reg, imp_mean.fit_transform(X_train), y_train, scoring='r2', cv=cv5)))

# %%
reg = SVR()
param_grid = {'C': [1e4, 1e5, 1e6, 1e7], 'gamma': [1e-10, 1e-9, 1e-8]}
rcv = GridSearchCV(reg, param_grid, n_jobs=-1, cv=cv5, scoring='r2')
rcv.fit(X_train, y_train)
print(rcv.best_score_)  # 0.255053624149075
print(rcv.best_params_)  # {'C': 100000.0, 'gamma': 1e-08}

# %%
