from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
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

data_file = configs['data_file']
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
cv = StratifiedKFold(n_splits=configs['cv'], shuffle=True, random_state=seed)

BATCH_SIZE = configs['batch_size']
NUM_WORKERS = 0
LOG_FREQ = configs['log_freq']
NUM_EPOCHS = configs['num_of_epochs']

def get_setup():
    if 'model' in configs and configs['model'] == 'MultiDrugResNN':
        model = MultiDrugResNN(X_train.shape[1], len(class_map_id2name),
                                configs['hidden_width'], configs['hidden_size'])
    else:
        model = MultiDrugResXNN(X_train.shape[1], len(class_map_id2name),
                                configs['hidden_width'], configs['hidden_size'], group=configs['group'])

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'])
    optimizer = RAdam(model.parameters(), lr=configs['lr'], weight_decay=configs['weight_decay'])
    logger.info(optimizer)

    lr_scheduler = None
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

data_sample = pd.read_csv(configs['data_file'], sep='\t')
logger.info(f"data matrix shape: {data_sample.shape}")

data_sample = data_sample.drop(['Cell_line'], axis=1)

imputer = get_imputer()
type_count = data_sample.groupby([configs['target']]).size()
selected_types = type_count[type_count > configs['cut_off']].index.values
data_sample = data_sample[data_sample[configs['target']].isin(selected_types)].reset_index(drop=True)
proteins = [x for x in data_sample.columns if '_HUMAN' in x]
X = data_sample[proteins]
y = data_sample[configs['target']]

if imputer:
    X = imputer.fit_transform(X)

class_map_name2id = dict(zip(sorted(set(y)), range(len(set(y)))))
class_map_id2name = dict(zip(range(len(set(y))), sorted(set(y))))
y = y.map(class_map_name2id)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=seed, shuffle=True,
                                                    test_size=0.2)
y_train = y_train.values
y_test = y_test.values

num_of_proteins = len(proteins)

val_score_dict = {'Drug Id': [], 'run': [], 'epoch': [], 'accuracy': []}

if configs['do_cv']:
    count = 0
    for cell_lines_train_index, cell_lines_val_index in cv.split(X_train, y_train):
        # impute NA
        X_train_cv = X_train[cell_lines_train_index, :]
        X_val_cv = X_train[cell_lines_val_index, :]
        y_train_cv = y_train[cell_lines_train_index]
        y_val_cv = y_train[cell_lines_val_index]

        train_dataset = ProteinDataset(X_train_cv, y_train_cv, mode='train', logger=logger)
        val_dataset = ProteinDataset(X_val_cv, y_val_cv, mode='val', logger=logger)

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

        train_res, val_res = train_loop_cls(NUM_EPOCHS, train_loader,
                                        val_loader, model, criterion,
                                        optimizer, logger, model_path, STAMP, configs,
                                        lr_scheduler, run=f"cv_{count}",
                                        val_score_dict=val_score_dict)
        count += 1

logger.info("CV finished. Now running full training")

train_dataset = ProteinDataset(X_train, y_train, mode='train', logger=logger)
test_dataset = ProteinDataset(X_test, y_test, mode='val', logger=logger)

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

train_res, test_res = train_loop_cls(NUM_EPOCHS, train_loader,
                                 test_loader, model, criterion,
                                 optimizer, logger, model_path, STAMP, configs,
                                 lr_scheduler, run=f"test",
                                 val_score_dict=val_score_dict)
if 'save_scores' not in configs or configs['save_scores']:
    val_score_df = pd.DataFrame(val_score_dict)
    val_score_df.to_csv(f"{configs['work_dir']}/scores_{STAMP}{log_suffix}.csv", index=False)
logger.info("Full training finished.")
