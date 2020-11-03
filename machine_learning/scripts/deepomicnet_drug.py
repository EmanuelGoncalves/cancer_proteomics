"""
Train DeepOmicNet for drug response data
python deepomicnet_drug.py configs/drug/protein/dl_ic50.json

"""
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import KFold
import logging
import sys
import json
import os
from datetime import datetime
import pandas as pd
from radam import RAdam
from torch.utils.data import DataLoader

from .deepomicnet_model import *

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

model_path = configs['model_path'] if 'model_path' in configs else ""
if not os.path.isdir(model_path):
    os.system(f"mkdir -p {model_path}")

meta_file = configs['meta_file']
data_file = configs['data_file']
cell_lines_train_file = configs['cell_lines_train']
cell_lines_test_file = configs['cell_lines_test']
ic50_file = configs['ic50_file']
data_type = configs['data_type']

log_file = f"{STAMP}{log_suffix}.log"
logger = logging.getLogger('multi-drug')
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

BATCH_SIZE = configs['batch_size']
NUM_WORKERS = 0
LOG_FREQ = configs['log_freq']
NUM_EPOCHS = configs['num_of_epochs']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_setup():
    if 'model' in configs and configs['model'] == 'DeepOmicNet':
        model = DeepOmicNet(train_df.shape[1], train_ic50.shape[1],
                                configs['hidden_width'], configs['hidden_size'])
    else:
        model = DeepOmicNetG(train_df.shape[1], train_ic50.shape[1],
                                configs['hidden_width'], configs['hidden_size'], group=configs['group'])
    logger.info(model)
    model = model.to(device)

    if 'loss' in configs and configs['loss'] == "corr":
        criterion = corr_loss
    elif 'loss' in configs and configs['loss'] == "smoothl1":
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'])
    optimizer = RAdam(model.parameters(), lr=configs['lr'], weight_decay=configs['weight_decay'])
    logger.info(optimizer)

    lr_scheduler = None
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                     T_max=configs['num_of_epochs'])
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 500, 800], gamma=0.1)
    return model, criterion, optimizer, lr_scheduler


def get_imputer():
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
    return imputer


meta = pd.read_csv(meta_file, sep='\t')
ic50 = pd.read_csv(ic50_file, low_memory=False)
min_cell_lines = configs['min_cell_lines']
ic50_counts = ic50.groupby(['drug_id']).size()
selected_drugs = ic50_counts[ic50_counts > min_cell_lines].index.values
logger.info(f"selected drugs {len(selected_drugs)}")

ic50_selected = ic50[ic50['drug_id'].isin(selected_drugs)]
ic50_selected_pivot = pd.pivot(ic50_selected[['cell_line_name', 'drug_id', configs['target']]], index='cell_line_name',
                               columns='drug_id', values=configs['target']).reset_index()

ic50_selected_pivot = ic50_selected_pivot.sort_values(by=['cell_line_name']).reset_index(drop=True)

if configs['data_type'] in ['protein', 'protein_rep',  'multiomic', 'peptide']:
    data_sample = pd.read_csv(data_file, sep='\t')
elif configs['data_type'] == 'rna':
    data_raw = pd.read_csv(data_file, index_col=0).T
    data_raw.index.name = 'SIDM'
    data_raw = data_raw.reset_index()
    data_sample = pd.merge(data_raw, meta[['SIDM', 'Cell_line']].drop_duplicates()).drop(['SIDM'],
                                                                                         axis=1)
    data_sample = data_sample.sort_values(by=['Cell_line'])
    data_sample = data_sample.set_index(['Cell_line'])
    data_sample = data_sample.reset_index()
elif configs['data_type'] == 'rna_common':
    proteins = pd.read_csv(cell_lines_train_file, sep='\t', index_col=0).columns
    data_raw = pd.read_csv(data_file, index_col=0).T
    data_raw.index.name = 'SIDM'
    data_raw = data_raw.reset_index()
    data_sample = pd.merge(data_raw, meta[['SIDM', 'Cell_line']].drop_duplicates()).drop(['SIDM'],
                                                                                         axis=1)
    data_sample = data_sample.sort_values(by=['Cell_line'])
    data_sample = data_sample.set_index(['Cell_line'])

    name_map = pd.read_csv("/home/scai/SangerDrug/data/misc/HUMAN_9606_idmapping.gene_prot.dat",
                           sep='\t',
                           names=['ID', 'type', 'code'])
    name_map = name_map.drop_duplicates(['ID', 'type'])
    name_map = pd.pivot(name_map, index='ID', columns='type', values='code').dropna()
    protein2rna_map = dict(zip(name_map['UniProtKB-ID'].values, name_map['Gene_Name'].values))
    rna2protein_map = {v: k for k, v in protein2rna_map.items()}
    genes = [protein2rna_map[x] for x in proteins]
    genes = list(set(genes).intersection(data_sample.columns))
    data_sample = data_sample[genes].reset_index()
elif configs['data_type'].lower() in ('wes', 'cna', 'methylation'):
    data_sample = pd.read_csv(data_file)
else:
    raise Exception

cell_lines_train = pd.read_csv(cell_lines_train_file, sep='\t')['Cell_line'].values
cell_lines_test = pd.read_csv(cell_lines_test_file, sep='\t')['Cell_line'].values

ic50_selected_pivot_train = ic50_selected_pivot[
    ic50_selected_pivot['cell_line_name'].isin(cell_lines_train)].reset_index(drop=True)
ic50_selected_pivot_test = ic50_selected_pivot[
    ic50_selected_pivot['cell_line_name'].isin(cell_lines_test)].reset_index(drop=True)

num_of_proteins = data_sample.shape[1] - 2 if configs['data_type'] == 'protein_rep' else data_sample.shape[1] - 1

logger.info(f"{len(cell_lines_train)} Training cell lines: {sorted(cell_lines_train)}")
logger.info(f"{len(cell_lines_test)} Testing cell lines: {sorted(cell_lines_test)}")

data_sample_train = data_sample[data_sample['Cell_line'].isin(cell_lines_train)].reset_index(drop=True)
data_sample_test = data_sample[data_sample['Cell_line'].isin(cell_lines_test)].reset_index(drop=True)

val_score_dict = {'drug_id': [], 'run': [], 'epoch': [], 'corr': [], 'mae': []}

if configs['do_cv']:
    count = 0
    for cell_lines_train_index, cell_lines_val_index in cv.split(cell_lines_train):

        # impute NA
        imputer = get_imputer()

        train_lines = np.array(cell_lines_train)[cell_lines_train_index]
        val_lines = np.array(cell_lines_train)[cell_lines_val_index]

        merged_df_train = pd.merge(data_sample_train[data_sample_train['Cell_line'].isin(train_lines)],
                                   ic50_selected_pivot_train, left_on=['Cell_line'],
                                   right_on=['cell_line_name']).drop(['cell_line_name'], axis=1)
        if configs['data_type'] != 'protein_rep':
            val_data = data_sample_train[data_sample_train['Cell_line'].isin(val_lines)]
        else:
            merged_df_train = merged_df_train.drop(['Automatic_MS_filename'], axis=1)
            val_data = data_sample_train[data_sample_train['Cell_line'].isin(val_lines)].drop(['Automatic_MS_filename'],
                                                                                              axis=1).groupby(
                ['Cell_line']).agg(np.nanmean).reset_index()

        merged_df_val = pd.merge(val_data,
                                 ic50_selected_pivot_train,
                                 left_on=['Cell_line'],
                                 right_on=['cell_line_name']).drop(['cell_line_name'], axis=1)

        train_ic50 = merged_df_train.iloc[:, (num_of_proteins + 1):]
        val_ic50 = merged_df_val.iloc[:, (num_of_proteins + 1):]
        train_df = merged_df_train.iloc[:, 1:(num_of_proteins + 1)]
        val_df = merged_df_val.iloc[:, 1:(num_of_proteins + 1)]

        val_drug_ids = merged_df_train.columns[(num_of_proteins + 1):]
        if imputer:
            X_train = imputer.fit_transform(train_df)
            X_val = imputer.transform(val_df)
        else:
            X_train = train_df
            X_val = val_df
        train_dataset = ProteinDataset(X_train, train_ic50, mode='train', logger=logger)
        val_dataset = ProteinDataset(X_val, val_ic50, mode='val', logger=logger)

        train_loader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  drop_last=configs['drop_last'],
                                  num_workers=NUM_WORKERS)

        val_loader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=NUM_WORKERS)

        model, criterion, optimizer, lr_scheduler = get_setup()

        train_res, val_res = train_loop(NUM_EPOCHS, train_loader,
                                        val_loader, model, criterion,
                                        optimizer, logger, model_path, STAMP, configs,
                                        lr_scheduler, val_drug_ids, run=f"cv_{count}",
                                        val_score_dict=val_score_dict)
        count += 1

logger.info("CV finished. Now running full training")
imputer = get_imputer()

merged_df_train = pd.merge(data_sample_train, ic50_selected_pivot_train, left_on=['Cell_line'],
                           right_on=['cell_line_name']).drop(['cell_line_name'], axis=1)

if configs['data_type'] != 'protein_rep':
    test_data = data_sample_test
else:
    merged_df_train = merged_df_train.drop(['Automatic_MS_filename'], axis=1)
    test_data = data_sample_test.drop(['Automatic_MS_filename'], axis=1).groupby(
        ['Cell_line']).agg(np.nanmean).reset_index()

merged_df_test = pd.merge(test_data, ic50_selected_pivot_test, left_on=['Cell_line'],
                          right_on=['cell_line_name']).drop(['cell_line_name'], axis=1)

train_df = merged_df_train.iloc[:, 1:(num_of_proteins + 1)]
train_ic50 = merged_df_train.iloc[:, (num_of_proteins + 1):]
test_ic50 = merged_df_test.iloc[:, (num_of_proteins + 1):]
test_df = merged_df_test.iloc[:, 1:(num_of_proteins + 1)]

if imputer:
    X_train = imputer.fit_transform(train_df)
    X_test = imputer.transform(test_df)
else:
    X_train = train_df
    X_test = test_df

train_dataset = ProteinDataset(X_train, train_ic50, mode='train', logger=logger)
test_dataset = ProteinDataset(X_test, test_ic50, mode='val', logger=logger)

train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          drop_last=configs['drop_last'],
                          num_workers=NUM_WORKERS)

test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS)

model, criterion, optimizer, lr_scheduler = get_setup()

val_drug_ids = merged_df_test.columns[(num_of_proteins + 1):]

train_res, test_res = train_loop(NUM_EPOCHS, train_loader,
                                 test_loader, model, criterion,
                                 optimizer, logger, model_path, STAMP, configs,
                                 lr_scheduler, val_drug_ids, run=f"test",
                                 val_score_dict=val_score_dict)
if 'save_scores' not in configs or configs['save_scores']:
    val_score_df = pd.DataFrame(val_score_dict)
    val_score_df.to_csv(f"{configs['work_dir']}/scores_{STAMP}{log_suffix}.csv", index=False)
logger.info("Full training finished.")
