import shap
import json

from multi_drug_model import *
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader
import pickle
from datetime import datetime

config_file = '/home/scai/SangerDrug/configs/protein/pytorch_scratch.json'
configs = json.load(open(config_file, 'r'))

meta_file = configs['meta_file']
data_file = configs['data_file']
ic50_file = configs['ic50_file']
data_type = configs['data_type']

seed = configs['seed']
cv = KFold(n_splits=configs['cv'], shuffle=True, random_state=seed)

BATCH_SIZE = configs['batch_size']
NUM_WORKERS = configs['num_workers']
LOG_FREQ = configs['log_freq']
NUM_EPOCHS = configs['num_of_epochs']

meta = pd.read_csv(meta_file, sep='\t')
ic50 = pd.read_csv(ic50_file)
ic50 = ic50.sort_values(by=['Dataset version'])
ic50 = ic50.drop_duplicates(
    ['Drug Id', 'Cell line name'], keep='last').sort_values(
    by=['Drug Id', 'Cell line name']).reset_index(drop=True)
min_cell_lines = configs['min_cell_lines']
ic50_counts = ic50.groupby(['Drug Id']).size()
selected_drugs = ic50_counts[ic50_counts > min_cell_lines].index.values

# selected_drugs = [257]

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

cell_lines_train, cell_lines_test = train_test_split(sorted(data_sample['Cell_line'].unique()),
                                                     test_size=0.2,
                                                     random_state=seed)

ic50_selected_pivot_train = ic50_selected_pivot[
    ic50_selected_pivot['Cell line name'].isin(cell_lines_train)].reset_index(drop=True)
ic50_selected_pivot_test = ic50_selected_pivot[
    ic50_selected_pivot['Cell line name'].isin(cell_lines_test)].reset_index(drop=True)

num_of_proteins = data_sample.shape[1] - 1

data_sample_train = data_sample[data_sample['Cell_line'].isin(cell_lines_train)].reset_index(drop=True)
data_sample_test = data_sample[data_sample['Cell_line'].isin(cell_lines_test)].reset_index(drop=True)

merged_df_train = pd.merge(data_sample_train, ic50_selected_pivot_train, left_on=['Cell_line'],
                           right_on=['Cell line name']).drop(['Cell line name'], axis=1)
merged_df_test = pd.merge(data_sample_test, ic50_selected_pivot_test, left_on=['Cell_line'],
                          right_on=['Cell line name']).drop(['Cell line name'], axis=1)

# load the model
model = MultiDrugResXNN(4020, 446, 4000, 1, group=2000)
model.load_state_dict(torch.load("/home/scai/SangerDrug/models/protein/MultiDrug/202003241143_test_140.pth"))

imputer = KNNImputer(missing_values=np.nan)

train_df = merged_df_train.iloc[:, 1:(num_of_proteins + 1)]
train_ic50 = merged_df_train.iloc[:, (num_of_proteins + 1):]
test_ic50 = merged_df_test.iloc[:, (num_of_proteins + 1):]
test_df = merged_df_test.iloc[:, 1:(num_of_proteins + 1)]

train_dataset = ProteinDataset(imputer.fit_transform(train_df), train_ic50, mode='train')
test_dataset = ProteinDataset(imputer.transform(test_df), test_ic50, mode='test')

train_loader = DataLoader(train_dataset,
                          batch_size=100,
                          shuffle=True,
                          drop_last=True,
                          num_workers=1)

test_loader = DataLoader(test_dataset,
                         batch_size=50,
                         shuffle=True,
                         num_workers=1)

# %%
X = next(iter(test_loader))

# %%
model = model.to('cuda')

drug_score = pd.read_csv("/home/scai/SangerDrug/data/drug/202003211831_drug_scores.tsv", sep="\t")
selected_drug_ids = drug_score[(drug_score['MultiDrug_correlation'] > 0.45) & (drug_score['sensitive_ratio'] == 1)][
    'Drug Id'].values

drug_columns_idx = np.array([test_ic50.columns.get_loc(x) for x in selected_drug_ids])
drug_columns_idx = np.tile(drug_columns_idx, (50, 1))

print("running GradientExplainer")
start = datetime.now()
e = shap.GradientExplainer((model, model.input), X[0].float().to('cuda'))
shap_values, indexes = e.shap_values(X[0].float().to('cuda'), nsamples=10, ranked_outputs=drug_columns_idx,
                                     output_rank_order='custom')
pickle.dump(shap_values, open('/home/scai/SangerDrug/work_dirs/shap/gradient_shap_50_20_selected.pkl', 'wb'))
pickle.dump(indexes, open('/home/scai/SangerDrug/work_dirs/shap/gradient_indexes_50_20_selected.pkl', 'wb'))
print(datetime.now() - start)

# print("running DeepExplainer")
# background = torch.rand(50, 4020) * 10 - 5
# e = shap.DeepExplainer(model, background.to('cuda'))
# start = datetime.now()
# shap_values = e.shap_values(X[0].float().to('cuda'))
# pickle.dump(shap_values, open('/home/scai/SangerDrug/work_dirs/shap/deep_shap.pkl', 'wb'))
# print(datetime.now()-start)
