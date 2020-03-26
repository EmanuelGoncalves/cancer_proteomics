from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import KFold
import logging
import sys
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
from radam import RAdam
from torch.utils.data import DataLoader

from multi_drug_model import *

STAMP = datetime.today().strftime('%Y%m%d%H%M')

config_file = sys.argv[1]

log_suffix = ''
if len(sys.argv) > 2:
    log_suffix = sys.argv[2]

# load model configs
configs = json.load(open(config_file, 'r'))
if not os.path.isdir(configs['work_dir']):
    os.system(f"mkdir -p {configs['work_dir']}")

model_path = configs['model_path']
if not os.path.isdir(model_path):
    os.system(f"mkdir -p {model_path}")

meta_file = configs['meta_file']
data_file = configs['data_file']
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

seed = configs['seed']
cv = KFold(n_splits=configs['cv'], shuffle=True, random_state=seed)

BATCH_SIZE = configs['batch_size']
NUM_WORKERS = configs['num_workers']
LOG_FREQ = configs['log_freq']
NUM_EPOCHS = configs['num_of_epochs']


def get_setup():
    # model = MultiDrugNN(train_df.shape[1], train_ic50.shape[1],
    #                     configs['hidden_width'], configs['hidden_size'])
    model = MultiDrugResXNN(train_df.shape[1], train_ic50.shape[1],
                            configs['hidden_width'], configs['hidden_size'], group=configs['group'])

    model = model.to(device)

    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'])
    optimizer = RAdam(model.parameters(), lr=configs['lr'])
    logger.info(optimizer)

    lr_scheduler = None
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                     T_max=configs['num_of_epochs'])
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 500, 800], gamma=0.1)
    return model, criterion, optimizer, lr_scheduler


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

val_score_dict = {'Drug Id': [], 'run': [], 'epoch': [], 'corr': [], 'mae': []}

if configs['do_cv']:
    count = 0
    for train_index, test_index in cv.split(merged_df_train):

        # impute NA
        imputer = None
        if configs['imputer'] == 'mean':
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        elif configs['imputer'] == 'KNN':
            imputer = KNNImputer(missing_values=np.nan)
        else:
            logger.warning("No imputer selected!")

        train_ic50 = merged_df_train.iloc[train_index, (num_of_proteins + 1):]
        val_ic50 = merged_df_train.iloc[test_index, (num_of_proteins + 1):]
        train_df = merged_df_train.iloc[train_index, 1:(num_of_proteins + 1)]
        val_df = merged_df_train.iloc[test_index, 1:(num_of_proteins + 1)]

        val_drug_ids = merged_df_train.columns[(num_of_proteins + 1):]
        train_dataset = ProteinDataset(imputer.fit_transform(train_df), train_ic50, mode='train', logger=logger)
        val_dataset = ProteinDataset(imputer.transform(val_df), val_ic50, mode='val', logger=logger)

        train_loader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=NUM_WORKERS)

        val_loader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=NUM_WORKERS)

        model, criterion, optimizer, lr_scheduler = get_setup()

        train_res, val_res = train_loop(NUM_EPOCHS, train_loader,
                                        val_loader, model, criterion,
                                        optimizer, logger, model_path, STAMP,
                                        lr_scheduler, val_drug_ids, run=f"cv_{count}",
                                        val_score_dict=val_score_dict)
        count += 1

logger.info("CV finished. Now running full training")
imputer = None
if configs['imputer'] == 'mean':
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
elif configs['imputer'] == 'KNN':
    imputer = KNNImputer(missing_values=np.nan)
else:
    logger.warning("No imputer selected!")

train_df = merged_df_train.iloc[:, 1:(num_of_proteins + 1)]
train_ic50 = merged_df_train.iloc[:, (num_of_proteins + 1):]
test_ic50 = merged_df_test.iloc[:, (num_of_proteins + 1):]
test_df = merged_df_test.iloc[:, 1:(num_of_proteins + 1)]

train_dataset = ProteinDataset(imputer.fit_transform(train_df), train_ic50, mode='train', logger=logger)
test_dataset = ProteinDataset(imputer.transform(test_df), test_ic50, mode='test', logger=logger)

train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          drop_last=True,
                          num_workers=NUM_WORKERS)

test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS)

model, criterion, optimizer, lr_scheduler = get_setup()

val_drug_ids = merged_df_test.columns[(num_of_proteins + 1):]

train_res, test_res = train_loop(NUM_EPOCHS, train_loader,
                                 test_loader, model, criterion,
                                 optimizer, logger, model_path, STAMP,
                                 lr_scheduler, val_drug_ids, run=f"test",
                                 val_score_dict=val_score_dict)

val_score_df = pd.DataFrame(val_score_dict)
val_score_df.to_csv(f"{configs['work_dir']}/scores_{STAMP}{log_suffix}.csv", index=False)
logger.info("Full training finished.")
