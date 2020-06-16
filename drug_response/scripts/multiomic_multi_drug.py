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
data_file_cna = configs['data_file_cna']
data_file_rna = configs['data_file_rna']
data_file_protein = configs['data_file_protein']
data_file_methylation = configs['data_file_methylation']
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

seed = configs['seed']
cv = KFold(n_splits=configs['cv'], shuffle=True, random_state=seed)

BATCH_SIZE = configs['batch_size']
NUM_WORKERS = 0
LOG_FREQ = configs['log_freq']
NUM_EPOCHS = configs['num_of_epochs']
NUM_OF_OMICS = 4 if configs['include_methy'] else 3


def get_setup():
    model = MultiOmicDrugResXNNV2(train_cna_df.shape[1], NUM_OF_OMICS, train_ic50.shape[1],
                                  configs['hidden_width'], configs['hidden_size'], group=configs['group'])
    logger.info(f"using {configs['model']}")
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
ic50_selected_pivot = pd.pivot(ic50_selected[['Cell line name', 'Drug Id', configs['target']]], index='Cell line name',
                               columns='Drug Id', values=configs['target']).reset_index()

ic50_selected_pivot = ic50_selected_pivot.sort_values(by=['Cell line name']).reset_index(drop=True)

data_sample_cna = pd.read_csv(data_file_cna, sep='\t')
data_sample_rna = pd.read_csv(data_file_rna, sep='\t')
data_sample_protein = pd.read_csv(data_file_protein, sep='\t')
data_sample_methylation = pd.read_csv(data_file_methylation, sep='\t')

cell_lines_train = pd.read_csv(cell_lines_train_file, sep='\t')['Cell_line'].values
cell_lines_test = pd.read_csv(cell_lines_test_file, sep='\t')['Cell_line'].values

ic50_selected_pivot_train = ic50_selected_pivot[
    ic50_selected_pivot['Cell line name'].isin(cell_lines_train)].reset_index(drop=True)
ic50_selected_pivot_test = ic50_selected_pivot[
    ic50_selected_pivot['Cell line name'].isin(cell_lines_test)].reset_index(drop=True)

num_of_proteins = data_sample_protein.shape[1] - 1

logger.info(f"{len(cell_lines_train)} Training cell lines: {sorted(cell_lines_train)}")
logger.info(f"{len(cell_lines_test)} Testing cell lines: {sorted(cell_lines_test)}")

data_sample_cna_train = data_sample_cna[data_sample_cna['Cell_line'].isin(cell_lines_train)].reset_index(drop=True)
data_sample_cna_test = data_sample_cna[data_sample_cna['Cell_line'].isin(cell_lines_test)].reset_index(drop=True)
data_sample_rna_train = data_sample_rna[data_sample_rna['Cell_line'].isin(cell_lines_train)].reset_index(drop=True)
data_sample_rna_test = data_sample_rna[data_sample_rna['Cell_line'].isin(cell_lines_test)].reset_index(drop=True)
data_sample_protein_train = data_sample_protein[data_sample_protein['Cell_line'].isin(cell_lines_train)].reset_index(
    drop=True)
data_sample_protein_test = data_sample_protein[data_sample_protein['Cell_line'].isin(cell_lines_test)].reset_index(
    drop=True)
data_sample_methylation_train = data_sample_methylation[
    data_sample_methylation['Cell_line'].isin(cell_lines_train)].reset_index(
    drop=True)
data_sample_methylation_test = data_sample_methylation[
    data_sample_methylation['Cell_line'].isin(cell_lines_test)].reset_index(
    drop=True)

val_score_dict = {'Drug Id': [], 'run': [], 'epoch': [], 'corr': [], 'mae': []}

if configs['do_cv']:
    count = 0
    for cell_lines_train_index, cell_lines_val_index in cv.split(cell_lines_train):
        train_lines = np.array(cell_lines_train)[cell_lines_train_index]
        val_lines = np.array(cell_lines_train)[cell_lines_val_index]

        merged_df_cna_train = pd.merge(data_sample_cna_train[data_sample_cna_train['Cell_line'].isin(train_lines)],
                                       ic50_selected_pivot_train, left_on=['Cell_line'],
                                       right_on=['Cell line name']).drop(['Cell line name'], axis=1)
        merged_df_rna_train = pd.merge(data_sample_rna_train[data_sample_rna_train['Cell_line'].isin(train_lines)],
                                       ic50_selected_pivot_train, left_on=['Cell_line'],
                                       right_on=['Cell line name']).drop(['Cell line name'], axis=1)
        merged_df_protein_train = pd.merge(
            data_sample_protein_train[data_sample_protein_train['Cell_line'].isin(train_lines)],
            ic50_selected_pivot_train, left_on=['Cell_line'],
            right_on=['Cell line name']).drop(['Cell line name'], axis=1)
        merged_df_methylation_train = pd.merge(
            data_sample_methylation_train[data_sample_methylation_train['Cell_line'].isin(train_lines)],
            ic50_selected_pivot_train, left_on=['Cell_line'],
            right_on=['Cell line name']).drop(['Cell line name'], axis=1)

        val_data_cna = data_sample_cna_train[data_sample_cna_train['Cell_line'].isin(val_lines)]
        val_data_rna = data_sample_rna_train[data_sample_rna_train['Cell_line'].isin(val_lines)]
        val_data_protein = data_sample_protein_train[data_sample_protein_train['Cell_line'].isin(val_lines)]
        val_data_methylation = data_sample_methylation_train[data_sample_methylation_train['Cell_line'].isin(val_lines)]

        merged_df_cna_val = pd.merge(val_data_cna,
                                     ic50_selected_pivot_train,
                                     left_on=['Cell_line'],
                                     right_on=['Cell line name']).drop(['Cell line name'], axis=1)
        merged_df_rna_val = pd.merge(val_data_rna,
                                     ic50_selected_pivot_train,
                                     left_on=['Cell_line'],
                                     right_on=['Cell line name']).drop(['Cell line name'], axis=1)
        merged_df_protein_val = pd.merge(val_data_protein,
                                         ic50_selected_pivot_train,
                                         left_on=['Cell_line'],
                                         right_on=['Cell line name']).drop(['Cell line name'], axis=1)
        merged_df_methylation_val = pd.merge(val_data_methylation,
                                             ic50_selected_pivot_train,
                                             left_on=['Cell_line'],
                                             right_on=['Cell line name']).drop(['Cell line name'], axis=1)

        train_ic50 = merged_df_cna_train.iloc[:, (num_of_proteins + 1):]
        val_ic50 = merged_df_cna_val.iloc[:, (num_of_proteins + 1):]
        train_cna_df = merged_df_cna_train.iloc[:, 1:(num_of_proteins + 1)]
        val_cna_df = merged_df_cna_val.iloc[:, 1:(num_of_proteins + 1)]
        train_rna_df = merged_df_rna_train.iloc[:, 1:(num_of_proteins + 1)]
        val_rna_df = merged_df_rna_val.iloc[:, 1:(num_of_proteins + 1)]
        train_protein_df = merged_df_protein_train.iloc[:, 1:(num_of_proteins + 1)]
        val_protein_df = merged_df_protein_val.iloc[:, 1:(num_of_proteins + 1)]
        train_methylation_df = merged_df_methylation_train.iloc[:, 1:(num_of_proteins + 1)]
        val_methylation_df = merged_df_methylation_val.iloc[:, 1:(num_of_proteins + 1)]

        val_drug_ids = merged_df_cna_val.columns[(num_of_proteins + 1):]

        X_train_cna = train_cna_df
        X_val_cna = val_cna_df
        X_train_rna = train_rna_df
        X_val_rna = val_rna_df
        X_train_protein = train_protein_df
        X_val_protein = val_protein_df
        X_train_methylation = train_methylation_df if configs['include_methy'] else None
        X_val_methylation = val_methylation_df if configs['include_methy'] else None

        train_dataset = MultiOmicDataset(X_train_cna, X_train_rna, X_train_protein, train_ic50, mode='train',
                                         methy_df=X_train_methylation, logger=logger)
        val_dataset = MultiOmicDataset(X_val_cna, X_val_rna, X_val_protein, val_ic50, mode='val',
                                       methy_df=X_val_methylation, logger=logger)

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

merged_df_cna_train = pd.merge(data_sample_cna_train,
                               ic50_selected_pivot_train, left_on=['Cell_line'],
                               right_on=['Cell line name']).drop(['Cell line name'], axis=1)
merged_df_rna_train = pd.merge(data_sample_rna_train,
                               ic50_selected_pivot_train, left_on=['Cell_line'],
                               right_on=['Cell line name']).drop(['Cell line name'], axis=1)
merged_df_protein_train = pd.merge(
    data_sample_protein_train,
    ic50_selected_pivot_train, left_on=['Cell_line'],
    right_on=['Cell line name']).drop(['Cell line name'], axis=1)
merged_df_methylation_train = pd.merge(
    data_sample_methylation_train,
    ic50_selected_pivot_train, left_on=['Cell_line'],
    right_on=['Cell line name']).drop(['Cell line name'], axis=1)

test_data_cna = data_sample_cna_test
test_data_rna = data_sample_rna_test
test_data_protein = data_sample_protein_test
test_data_methylation = data_sample_methylation_test

merged_df_cna_test = pd.merge(test_data_cna, ic50_selected_pivot_test, left_on=['Cell_line'],
                              right_on=['Cell line name']).drop(['Cell line name'], axis=1)
merged_df_rna_test = pd.merge(test_data_rna, ic50_selected_pivot_test, left_on=['Cell_line'],
                              right_on=['Cell line name']).drop(['Cell line name'], axis=1)
merged_df_protein_test = pd.merge(test_data_protein, ic50_selected_pivot_test, left_on=['Cell_line'],
                                  right_on=['Cell line name']).drop(['Cell line name'], axis=1)
merged_df_methylation_test = pd.merge(test_data_methylation, ic50_selected_pivot_test, left_on=['Cell_line'],
                                      right_on=['Cell line name']).drop(['Cell line name'], axis=1)

train_cna_df = merged_df_cna_train.iloc[:, 1:(num_of_proteins + 1)]
train_rna_df = merged_df_rna_train.iloc[:, 1:(num_of_proteins + 1)]
train_protein_df = merged_df_protein_train.iloc[:, 1:(num_of_proteins + 1)]
train_methylation_df = merged_df_methylation_train.iloc[:, 1:(num_of_proteins + 1)] if configs[
    'include_methy'] else None

test_cna_df = merged_df_cna_test.iloc[:, 1:(num_of_proteins + 1)]
test_rna_df = merged_df_rna_test.iloc[:, 1:(num_of_proteins + 1)]
test_protein_df = merged_df_protein_test.iloc[:, 1:(num_of_proteins + 1)]
test_methylation_df = merged_df_methylation_test.iloc[:, 1:(num_of_proteins + 1)] if configs['include_methy'] else None

train_ic50 = merged_df_cna_train.iloc[:, (num_of_proteins + 1):]
test_ic50 = merged_df_cna_test.iloc[:, (num_of_proteins + 1):]

train_dataset = MultiOmicDataset(train_cna_df, train_rna_df, train_protein_df, train_ic50, mode='train',
                                 methy_df=train_methylation_df, logger=logger)
test_dataset = MultiOmicDataset(test_cna_df, test_rna_df, test_protein_df, test_ic50, mode='val',
                                methy_df=test_methylation_df, logger=logger)

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

val_drug_ids = merged_df_cna_test.columns[(num_of_proteins + 1):]

train_res, test_res = train_loop(NUM_EPOCHS, train_loader,
                                 test_loader, model, criterion,
                                 optimizer, logger, model_path, STAMP, configs,
                                 lr_scheduler, val_drug_ids, run=f"test",
                                 val_score_dict=val_score_dict)
if configs['save_scores']:
    val_score_df = pd.DataFrame(val_score_dict)
    val_score_df.to_csv(f"{configs['work_dir']}/scores_{STAMP}{log_suffix}.csv", index=False)

logger.info("Full training finished.")
