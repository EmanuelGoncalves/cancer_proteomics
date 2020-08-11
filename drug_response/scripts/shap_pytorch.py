import shap
import json

from multi_drug_model import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader
import pickle
import torch
from joblib import Parallel, delayed
from datetime import datetime
import sys
import logging
import os

torch.backends.cudnn.benchmark = True

config_file = sys.argv[1]
configs = json.load(open(config_file, 'r'))

STAMP = datetime.today().strftime('%Y%m%d%H%M')
log_suffix = ''
if 'suffix' in configs:
    log_suffix = configs['suffix']
elif len(sys.argv) > 2:
    log_suffix = sys.argv[2]
if not os.path.isdir(configs['shape_work_dir']):
    os.system(f"mkdir -p {configs['shape_work_dir']}")

log_file = f"{STAMP}{log_suffix}_shap.log"
logger = logging.getLogger('multi-drug')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(configs["shape_work_dir"], log_file))
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# logger.addHandler(ch)
logger.addHandler(fh)

logger.info(open(config_file, 'r').read())

meta_file = configs['meta_file']
data_file = configs['data_file']
ic50_file = configs['ic50_file']
data_type = configs['data_type']
cell_lines_train_file = configs['cell_lines_train']
cell_lines_test_file = configs['cell_lines_test']
seed = configs['seed']

BATCH_SIZE = configs['batch_size']
NUM_WORKERS = configs['num_workers']
LOG_FREQ = configs['log_freq']
NUM_EPOCHS = configs['num_of_epochs']

meta = pd.read_csv(meta_file, sep='\t')
ic50 = pd.read_csv(ic50_file, low_memory=False)

min_cell_lines = configs['min_cell_lines']
ic50_counts = ic50.groupby(['drug_id']).size()
selected_drugs = ic50_counts[ic50_counts > min_cell_lines].index.values

# selected_drugs = [257]

ic50_selected = ic50[ic50['drug_id'].isin(selected_drugs)]
ic50_selected_pivot = pd.pivot(ic50_selected[['cell_line_name', 'drug_id', configs['target']]], index='cell_line_name',
                               columns='drug_id', values=configs['target']).reset_index()

ic50_selected_pivot = ic50_selected_pivot.sort_values(by=['cell_line_name']).reset_index(drop=True)

data_sample = pd.read_csv(data_file, sep='\t')

cell_lines_train = pd.read_csv(cell_lines_train_file, sep='\t')['Cell_line'].values
cell_lines_test = pd.read_csv(cell_lines_test_file, sep='\t')['Cell_line'].values

ic50_selected_pivot_train = ic50_selected_pivot[
    ic50_selected_pivot['cell_line_name'].isin(cell_lines_train)].reset_index(drop=True)
ic50_selected_pivot_test = ic50_selected_pivot[
    ic50_selected_pivot['cell_line_name'].isin(cell_lines_test)].reset_index(drop=True)

num_of_proteins = data_sample.shape[1] - 2 if configs['data_type'] == 'protein_rep' else data_sample.shape[1] - 1

data_sample_train = data_sample[data_sample['Cell_line'].isin(cell_lines_train)].reset_index(drop=True)
data_sample_test = data_sample[data_sample['Cell_line'].isin(cell_lines_test)].reset_index(drop=True)

merged_df_train = pd.merge(data_sample_train, ic50_selected_pivot_train, left_on=['Cell_line'],
                           right_on=['cell_line_name']).drop(['cell_line_name'], axis=1)

test_data = data_sample_test
merged_df_test = pd.merge(test_data, ic50_selected_pivot_test, left_on=['Cell_line'],
                          right_on=['cell_line_name']).drop(['cell_line_name'], axis=1)

train_df = merged_df_train.iloc[:, 1:(num_of_proteins + 1)]
train_ic50 = merged_df_train.iloc[:, (num_of_proteins + 1):]
test_ic50 = merged_df_test.iloc[:, (num_of_proteins + 1):]
test_df = merged_df_test.iloc[:, 1:(num_of_proteins + 1)]
X_train = train_df
X_test = test_df
train_dataset = ProteinDataset(X_train, train_ic50, mode='train')
test_dataset = ProteinDataset(X_test, test_ic50, mode='val')

merged_df_all = pd.merge(data_sample, ic50_selected_pivot, left_on=['Cell_line'],
                         right_on=['cell_line_name']).drop(['cell_line_name'], axis=1)
all_df = merged_df_all.iloc[:, 1:(num_of_proteins + 1)]
all_ic50 = merged_df_all.iloc[:, (num_of_proteins + 1):]
all_dataset = ProteinDataset(all_df, all_ic50, mode='test')

NUM_OF_SAMPLES = configs['shap_num_of_samples']
NUM_THREAD = 1

train_loader = DataLoader(train_dataset,
                          batch_size=NUM_OF_SAMPLES // NUM_THREAD,
                          shuffle=False,
                          num_workers=0)

test_loader = DataLoader(test_dataset,
                         batch_size=NUM_OF_SAMPLES // NUM_THREAD,
                         shuffle=False,
                         num_workers=0)

all_loader = DataLoader(all_dataset,
                        batch_size=NUM_OF_SAMPLES // NUM_THREAD,
                        shuffle=False,
                        num_workers=0)


# %%
def run_shap_gradient(data, drug_columns_idx, i, filename, N_SAMPLES):
    X, model = data
    e = shap.GradientExplainer((model, model.input), X[0].float().to('cuda'))
    shap_values, indexes = e.shap_values(X[0].float().to('cuda'), nsamples=N_SAMPLES, ranked_outputs=drug_columns_idx,
                                         output_rank_order='custom')
    # shap_values, indexes = e.shap_values(X[0].float().to('cuda'), nsamples=N_SAMPLES, ranked_outputs=2,
    #                                      output_rank_order='max_abs')
    # indexes = indexes.detach().cpu().numpy()
    pickle.dump(shap_values, open(f'{configs["shape_work_dir"]}/{filename}_shap_{i}.pkl', 'wb'))
    pickle.dump(indexes, open(f'{configs["shape_work_dir"]}/{filename}_indexes_{i}.pkl', 'wb'))


# %%
data = []
N_SAMPLES = configs['shap_nsamples']
it = iter(test_loader)
for i in range(NUM_THREAD):
    model = MultiDrugResXNN(num_of_proteins, len(selected_drugs), configs['hidden_width'], configs['hidden_size'],
                            group=configs['group'])
    model.load_state_dict(torch.load(
        f"{configs['model_path']}/{configs['shap_model']}.pth"))

    model = model.to('cuda')
    data.append((next(it), model))

drug_columns_idx = None
drug_score = pd.read_csv("/home/scai/SangerDrug/data/drug/final_drug_scores_eg_id.tsv", sep="\t")
selected_drug_ids = drug_score[
    (drug_score['MultiDrug_correlation'] > configs['shap_min_corr']) & (
                drug_score['sensitive_count'] > configs['shap_min_sensitive'])][
    'drug_id'].values

# selected_drug_ids = sorted([1909, 1114])

drug_columns_idx = np.array([train_ic50.columns.get_loc(x) for x in selected_drug_ids])
drug_columns_idx = np.tile(drug_columns_idx, (NUM_OF_SAMPLES // NUM_THREAD, 1))

logger.info("running GradientExplainer")
filename = f"gradient_{configs['shap_model']}_{STAMP}"
logger.info(filename)

Parallel(n_jobs=NUM_THREAD)(
    delayed(run_shap_gradient)(data[i], drug_columns_idx, i, filename, N_SAMPLES) for i in range(NUM_THREAD))

