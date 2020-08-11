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
from sklearn.preprocessing import MinMaxScaler

from multi_drug_model import *

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

model_path = configs['model_path']
if not os.path.isdir(model_path):
    os.system(f"mkdir -p {model_path}")

meta_file = configs['meta_file']
data_file = configs['data_file']
cell_lines_train_file = configs['cell_lines_train']
cell_lines_test_file = configs['cell_lines_test']
crispr_file = configs['crispr_file']
data_type = configs['data_type']

log_file = f"{STAMP}{log_suffix}.log"
logger = logging.getLogger('multi-crispr')
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
NUM_WORKERS = configs['num_workers']
LOG_FREQ = configs['log_freq']
NUM_EPOCHS = configs['num_of_epochs']
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_setup():
    if configs['model'] == 'MultiDrugResNN':
        model = MultiDrugResNN(train_df.shape[1], train_crispr.shape[1],
                               configs['hidden_width'], configs['hidden_size'])
    elif configs['model'] == 'MultiDrugResXNN':
        model = MultiDrugResXNN(train_df.shape[1], train_crispr.shape[1],
                                configs['hidden_width'], configs['hidden_size'], group=configs['group'])

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
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.2)
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
crispr = pd.read_csv(crispr_file)
crispr = crispr.melt(id_vars='Gene', var_name='Cell line name', value_name='logFC')
# crispr['logFC'] = MinMaxScaler().fit_transform(crispr['logFC'].values.reshape(-1, 1))

min_cell_lines = configs['min_cell_lines']
crispr_counts = crispr.groupby(['Gene']).size()
selected_genes = crispr_counts[crispr_counts > min_cell_lines].index.values

# selected_drugs = [257]

crispr_selected = crispr[crispr['Gene'].isin(selected_genes)]
crispr_selected_pivot = pd.pivot(crispr_selected[['Cell line name', 'Gene', configs['target']]], index='Cell line name',
                                 columns='Gene', values=configs['target']).reset_index()

crispr_selected_pivot = crispr_selected_pivot.sort_values(by=['Cell line name']).reset_index(drop=True)

if configs['data_type'] in ['protein', 'protein_rep', 'rna_common', 'multiomic', 'peptide', 'mofa']:
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
else:
    raise Exception

# data_sample = data_sample.set_index("Cell_line")
# data_sample = pd.DataFrame(MinMaxScaler().fit_transform(data_sample), index=data_sample.index)
# data_sample = data_sample.reset_index()

cell_lines_train = pd.read_csv(cell_lines_train_file, sep='\t')['Cell_line'].values
cell_lines_test = pd.read_csv(cell_lines_test_file, sep='\t')['Cell_line'].values

crispr_selected_pivot_train = crispr_selected_pivot[
    crispr_selected_pivot['Cell line name'].isin(cell_lines_train)].reset_index(drop=True)
crispr_selected_pivot_test = crispr_selected_pivot[
    crispr_selected_pivot['Cell line name'].isin(cell_lines_test)].reset_index(drop=True)

num_of_proteins = data_sample.shape[1] - 1

logger.info(f"{len(cell_lines_train)} Training cell lines: {sorted(cell_lines_train)}")
logger.info(f"{len(cell_lines_test)} Testing cell lines: {sorted(cell_lines_test)}")

data_sample_train = data_sample[data_sample['Cell_line'].isin(cell_lines_train)].reset_index(drop=True)
data_sample_test = data_sample[data_sample['Cell_line'].isin(cell_lines_test)].reset_index(drop=True)

val_score_dict = {'Gene': [], 'run': [], 'epoch': [], 'corr': [], 'mae': []}

if configs['do_cv']:
    count = 0
    for cell_lines_train_index, cell_lines_val_index in cv.split(cell_lines_train):

        # impute NA
        imputer = get_imputer()

        train_lines = np.array(cell_lines_train)[cell_lines_train_index]
        val_lines = np.array(cell_lines_train)[cell_lines_val_index]

        merged_df_train = pd.merge(data_sample_train[data_sample_train['Cell_line'].isin(train_lines)],
                                   crispr_selected_pivot_train, left_on=['Cell_line'],
                                   right_on=['Cell line name']).drop(['Cell line name'], axis=1)
        val_data = data_sample_train[data_sample_train['Cell_line'].isin(val_lines)]

        merged_df_val = pd.merge(val_data,
                                 crispr_selected_pivot_train,
                                 left_on=['Cell_line'],
                                 right_on=['Cell line name']).drop(['Cell line name'], axis=1)

        train_crispr = merged_df_train.iloc[:, (num_of_proteins + 1):]
        val_crispr = merged_df_val.iloc[:, (num_of_proteins + 1):]
        train_df = merged_df_train.iloc[:, 1:(num_of_proteins + 1)]
        val_df = merged_df_val.iloc[:, 1:(num_of_proteins + 1)]

        val_crispr_genes = merged_df_train.columns[(num_of_proteins + 1):]
        if imputer:
            X_train = imputer.fit_transform(train_df)
            X_val = imputer.transform(val_df)
        else:
            X_train = train_df
            X_val = val_df
        train_dataset = ProteinDataset(X_train, train_crispr, mode='train', logger=logger)
        val_dataset = ProteinDataset(X_val, val_crispr, mode='val', logger=logger)

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
                                        lr_scheduler, val_crispr_genes, run=f"cv_{count}",
                                        val_score_dict=val_score_dict)
        count += 1

logger.info("CV finished. Now running full training")
imputer = get_imputer()

merged_df_train = pd.merge(data_sample_train, crispr_selected_pivot_train, left_on=['Cell_line'],
                           right_on=['Cell line name']).drop(['Cell line name'], axis=1)

if configs['data_type'] != 'protein_rep':
    test_data = data_sample_test
else:
    merged_df_train = merged_df_train.drop(['Automatic_MS_filename'], axis=1)
    test_data = data_sample_test.drop(['Automatic_MS_filename'], axis=1).groupby(
        ['Cell_line']).agg(np.nanmean).reset_index()

merged_df_test = pd.merge(test_data, crispr_selected_pivot_test, left_on=['Cell_line'],
                          right_on=['Cell line name']).drop(['Cell line name'], axis=1)

train_df = merged_df_train.iloc[:, 1:(num_of_proteins + 1)]
train_crispr = merged_df_train.iloc[:, (num_of_proteins + 1):]
test_crispr = merged_df_test.iloc[:, (num_of_proteins + 1):]
test_df = merged_df_test.iloc[:, 1:(num_of_proteins + 1)]

if imputer:
    X_train = imputer.fit_transform(train_df)
    X_test = imputer.transform(test_df)
else:
    X_train = train_df
    X_test = test_df

train_dataset = ProteinDataset(X_train, train_crispr, mode='train', logger=logger)
test_dataset = ProteinDataset(X_test, test_crispr, mode='val', logger=logger)

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

val_crispr_genes = merged_df_test.columns[(num_of_proteins + 1):]

train_res, test_res = train_loop(NUM_EPOCHS, train_loader,
                                 test_loader, model, criterion,
                                 optimizer, logger, model_path, STAMP, configs,
                                 lr_scheduler, val_crispr_genes, run=f"test",
                                 val_score_dict=val_score_dict)
if 'save_scores' not in configs or configs['save_scores']:
    val_score_df = pd.DataFrame(val_score_dict)
    val_score_df.to_csv(f"{configs['work_dir']}/scores_{STAMP}{log_suffix}.csv", index=False)
logger.info("Full training finished.")
