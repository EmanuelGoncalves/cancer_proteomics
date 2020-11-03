"""
This file contains utility scripts and model architectures
"""
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from tqdm import trange
from scipy.stats import pearsonr
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
IGNORE_INDEX = -100


def logistic(x):
    return 1 / (1 + torch.exp(-x))


def corr_loss(output, target):
    x = output
    y = target

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    loss = 50 * (1 - torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))))

    return loss


class MultiDrugNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_width, hidden_size=4):
        super(MultiDrugNN, self).__init__()
        self.hidden = nn.ModuleList()

        self.input = nn.Linear(in_dim, hidden_width)
        for k in range(hidden_size):
            self.hidden.append(nn.Linear(hidden_width, hidden_width))
        self.output = nn.Linear(hidden_width, out_dim)

    def forward(self, x):
        activation = logistic
        x = activation(self.input(x))
        for layer in self.hidden:
            x = activation(layer(x))
        x = self.output(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, hidden_width, module_length=2):
        super(BasicBlock, self).__init__()
        self.hidden = nn.ModuleList()
        for k in range(module_length):
            self.hidden.append(
                nn.Sequential(nn.Linear(hidden_width, hidden_width)
                              )
            )

    def forward(self, x):
        activation = torch.relu
        identity = x
        out = x
        for layer in self.hidden:
            out = activation(layer(out))
        out += identity

        return out


class BottleNeck(nn.Module):
    def __init__(self, hidden_width, group=20):
        super(BottleNeck, self).__init__()
        self.groups = nn.ModuleList()
        for g in range(group):
            group_layers = nn.ModuleList()
            group_layers.append(
                nn.Sequential(nn.Linear(hidden_width, hidden_width // group)
                              ))
            group_layers.append(
                nn.Sequential(nn.Linear(hidden_width // group, hidden_width // group)
                              ))
            self.groups.append(group_layers)

    def forward(self, x):
        activation = logistic
        identity = x
        out = []
        for group_layers in self.groups:
            group_out = x
            for layer in group_layers:
                group_out = activation(layer(group_out))

            out.append(group_out)
        out = torch.cat(out, dim=1)
        out += identity
        return out


class DeepOmicNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_width, hidden_size=2):
        super(DeepOmicNet, self).__init__()

        self.input = nn.Linear(in_dim, hidden_width)
        self.hidden = nn.ModuleList()
        for k in range(hidden_size):
            self.hidden.append(BasicBlock(hidden_width))

        self.output = nn.Linear(hidden_width, out_dim)

    def forward(self, x):
        activation = logistic
        x = activation(self.input(x))
        for layer in self.hidden:
            x = activation(layer(x))
        x = self.output(x)
        return x


class DeepOmicNetG(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_width, hidden_size=2, group=20):
        super(DeepOmicNetG, self).__init__()

        self.input = nn.Linear(in_dim, hidden_width)
        self.hidden = nn.ModuleList()
        for k in range(hidden_size):
            self.hidden.append(BottleNeck(hidden_width, group=group))

        self.output = nn.Linear(hidden_width, out_dim)

    def forward(self, x):
        activation = logistic
        x = activation(self.input(x))
        for layer in self.hidden:
            x = activation(layer(x))
        x = self.output(x)
        return x


class ProteinDataset(Dataset):
    def __init__(self, data_df, purpose_data_df, mode, logger=None):
        assert mode in ['train', 'val', 'test']

        self.df = np.nan_to_num(data_df, nan=0)
        self.purpose_data = np.nan_to_num(purpose_data_df, nan=IGNORE_INDEX)

        assert self.df.shape[0] == self.purpose_data.shape[0], f"{self.df.shape[0]}, {self.purpose_data.shape[0]}"
        self.mode = mode
        if logger:
            logger.info(f"mode: {mode}, df shape: {self.df.shape}, purpose_data shape: {self.purpose_data.shape}")
        else:
            print(f"mode: {mode}, df shape: {self.df.shape}, purpose_data shape: {self.purpose_data.shape}")

    def __getitem__(self, index):
        """ Returns: tuple (sample, target) """
        data = self.df[index, :]
        if len(self.purpose_data.shape) > 1:
            target = self.purpose_data[index, :]  # the first col is cell line name
        else:
            target = self.purpose_data[index]

        # no other preprocessing for now

        return data, target

    def __len__(self):
        return self.df.shape[0]


class AverageMeter:
    ''' Computes and stores the average and current value '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_r2 = AverageMeter()
    avg_mae = AverageMeter()
    avg_rmse = AverageMeter()
    avg_corr = AverageMeter()

    model.train()
    num_steps = len(train_loader)

    end = time.time()
    lr_str = ''

    for i, (input_, targets) in enumerate(train_loader):
        if i >= num_steps:
            break

        output = model(input_.float().to(device))
        output[targets == IGNORE_INDEX] = IGNORE_INDEX

        loss = criterion(output, targets.float().to(device))
        targets = targets.cpu().numpy()

        confs = output.detach().cpu().numpy()
        if not np.isinf(confs).any() and not np.isnan(confs).any():
            try:
                avg_r2.update(np.median(
                    [r2_score(targets[targets[:, i] != IGNORE_INDEX, i], confs[targets[:, i] != IGNORE_INDEX, i])
                     for i in range(confs.shape[1])]))
                avg_mae.update(np.median(
                    [mean_absolute_error(targets[targets[:, i] != IGNORE_INDEX, i], confs[targets[:, i] != IGNORE_INDEX, i])
                     for i in range(confs.shape[1])]))
                avg_rmse.update(np.median(
                    [mean_squared_error(targets[targets[:, i] != IGNORE_INDEX, i], confs[targets[:, i] != IGNORE_INDEX, i],
                                        squared=True)
                     for i in range(confs.shape[1])]))
                avg_corr.update(np.median(
                    [pearsonr(targets[targets[:, i] != IGNORE_INDEX, i], confs[targets[:, i] != IGNORE_INDEX, i])[0]
                     for i in range(confs.shape[1])][0]))
            except ValueError:
                logger.info("skipping training score")

        losses.update(loss.data.item(), input_.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    logger.info(f'{epoch} \t'
                f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'corr {avg_corr.val:.4f} ({avg_corr.avg:.4f})\t'
                f'R2 {avg_r2.val:.4f} ({avg_r2.avg:.4f})\t'
                f'MAE {avg_mae.val:.4f} ({avg_mae.avg:.4f})\t'
                f'RMSE {avg_rmse.val:.4f} ({avg_rmse.avg:.4f})\t' + lr_str)

    return avg_r2.avg


def inference(data_loader, model):
    ''' Returns predictions and targets, if any. '''
    model.eval()

    all_confs, all_targets = [], []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            input_, target = data

            output = model(input_.float().to(device))
            all_confs.append(output)

            if target is not None:
                all_targets.append(target)

    confs = torch.cat(all_confs)
    targets = torch.cat(all_targets) if len(all_targets) else None
    targets = targets.cpu().numpy()
    confs = confs.cpu().numpy()

    return confs, targets


def validate(val_loader, model, val_drug_ids, run=None, epoch=None, val_score_dict=None):
    confs, targets = inference(val_loader, model)

    r2_avg, mae_avg, rmse_avg, corr_avg = None, None, None, None
    if not np.isinf(confs).any() and not np.isnan(confs).any():
        if 'drug_id' in val_score_dict:
            val_score_dict['drug_id'].extend(val_drug_ids)
        elif 'Gene' in val_score_dict:
            val_score_dict['Gene'].extend(val_drug_ids)
        else:
            raise
        val_score_dict['run'].extend([run] * len(val_drug_ids))
        val_score_dict['epoch'].extend([epoch] * len(val_drug_ids))

        r2 = [r2_score(targets[targets[:, i] != IGNORE_INDEX, i], confs[targets[:, i] != IGNORE_INDEX, i])
              for i in range(confs.shape[1])]
        r2_avg = np.median(r2)

        mae = [mean_absolute_error(targets[targets[:, i] != IGNORE_INDEX, i], confs[targets[:, i] != IGNORE_INDEX, i])
               for i in range(confs.shape[1])]
        mae_avg = np.median(mae)

        rmse = [mean_squared_error(targets[targets[:, i] != IGNORE_INDEX, i], confs[targets[:, i] != IGNORE_INDEX, i],
                                   squared=False)
                for i in range(confs.shape[1])]
        rmse_avg = np.median(rmse)

        corr = [pearsonr(targets[targets[:, i] != IGNORE_INDEX, i], confs[targets[:, i] != IGNORE_INDEX, i])[0]
                for i in range(confs.shape[1])]
        corr_avg = np.median(corr)

        val_score_dict['corr'].extend(corr)
        val_score_dict['mae'].extend(mae)

    return r2_avg, mae_avg, rmse_avg, corr_avg


def train_loop(epochs, train_loader, val_loader, model, criterion, optimizer, logger, model_path, stamp,
               configs,
               lr_scheduler=None,
               val_drug_ids=None,
               run=None, val_score_dict=None):
    train_res = []
    val_res = []
    for epoch in trange(1, epochs + 1):
        if lr_scheduler:
            logger.info(f"learning rate: {lr_scheduler.get_last_lr()}")
        train_score = train(train_loader,
                            model,
                            criterion,
                            optimizer,
                            epoch,
                            logger)

        train_res.append(train_score)
        if lr_scheduler:
            lr_scheduler.step()

        if val_loader:
            r2, mae, rmse, corr = validate(val_loader, model, val_drug_ids, run=run, epoch=epoch,
                                           val_score_dict=val_score_dict)
            if r2 and mae and rmse and corr:
                logger.info(f"Epoch {epoch} validation corr:{corr:4f}, R2:{r2:4f}, MAE:{mae:4f}, RMSE:{rmse:4f}")
            else:
                logger.info(f"Epoch {epoch} validation Inf")

        if configs['save_checkpoints'] and configs['check_start'] < epoch < configs['check_end']:
            torch.save(model.state_dict(), f"{model_path}/{stamp}{configs['suffix']}_{run}_{epoch}.pth")

    return np.asarray(train_res), np.asarray(val_res)
