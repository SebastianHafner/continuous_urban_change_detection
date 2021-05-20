import sys
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

from tabulate import tabulate
import wandb

from utils import datasets, metrics

from networks.network_loader import create_network, save_checkpoint
from networks import gru

from experiment_manager.args import default_argument_parser
from experiment_manager.config import config

from tqdm import tqdm


def run_training(cfg):

    run_config = {
        'CONFIG_NAME': cfg.NAME,
        'device': device,
        'epochs': cfg.TRAINER.EPOCHS,
        'learning rate': cfg.TRAINER.LR,
        'batch size': cfg.TRAINER.BATCH_SIZE,
    }
    table = {'run config name': run_config.keys(),
             ' ': run_config.values(),
             }
    print(tabulate(table, headers='keys', tablefmt="fancy_grid", ))

    batch_size = cfg.TRAINER.BATCH_SIZE
    in_size = cfg.MODEL.IN_CHANNELS  # number of features
    classes_no = cfg.MODEL.OUT_CHANNELS

    # loading network
    model = gru.GRUNet(input_dim=in_size, hidden_dim=10, output_dim=classes_no, n_layers=2)
    # model = nn.GRU(in_size, classes_no, 2) if cfg.MODEL.TYPE == 'gru' else nn.LSTM(in_size, classes_no, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    # reset the generators
    dataset = datasets.TrainingDataset(cfg=cfg, run_type='training')
    print(dataset)

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle':cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    save_checkpoints = cfg.CHECKPOINTS.SAVE
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        loss_set = []

        h0 = model.init_hidden(batch_size).to(device)

        for i, batch in enumerate(dataloader):

            model.train()
            optimizer.zero_grad()

            x = batch['x'].to(device)
            y_gts = batch['y'].to(device)

            y_pred, _ = model(x, h0)

            loss = criterion(y_pred, y_gts[:, 0].long())
            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            model_evaluation(model, cfg, device, 'training', epoch_float, global_step, max_samples=1_000)

            if global_step % cfg.LOG_FREQ == 0 and not cfg.DEBUG:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')

                # evaluation on sample of training and validation set
                model_evaluation(model, cfg, device, 'training', epoch_float, global_step, max_samples=1_000)
                model_evaluation(model, cfg, device, 'validation', epoch_float, global_step, max_samples=1_000)

                # logging
                time = timeit.default_timer() - start
                wandb.log({
                    'loss': np.mean(loss_set),
                    'labeled_percentage': 100,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_set = []

            if cfg.DEBUG:
                break
            # end of batch

        if not cfg.DEBUG:
            assert(epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        if epoch in save_checkpoints and not cfg.DEBUG:
            print(f'saving network', flush=True)
            save_checkpoint(model, optimizer, epoch, global_step, cfg)


def model_evaluation(model, cfg, device: str, run_type: str, epoch_float: float, global_step: int,
                     max_samples: int = 1_000):

    y_gts, y_preds = [], []

    model.to(device)
    model.eval()

    dataset = datasets.InferenceDataset(cfg, run_type)

    h0 = model.init_hidden(4).to(device)
    sm = torch.nn.Softmax(dim=1)
    for index in range(len(dataset)):

        patch = dataset.__getitem__(index)
        x = patch['x'].to(device)
        y_gt = patch['y'].to(device)

        y_pred, _ = model(x, h0)
        y_pred = sm(y_pred)
        y_pred = torch.argmax(y_pred, dim=1)

        y_gts.append(y_gt.flatten().cpu().numpy())
        y_preds.append(y_pred.flatten().cpu().numpy())

    y_gts, y_preds = np.concatenate(y_gts), np.concatenate(y_preds)
    f1 = metrics.compute_f1_score(y_preds, y_gts)
    p = metrics.compute_precision(y_preds, y_gts)
    r = metrics.compute_recall(y_preds, y_gts)



if __name__ == '__main__':

    args = default_argument_parser().parse_known_args()[0]
    cfg = config.setup(args)

    # make training deterministic
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cudnn.benchmark = True # faster convolutions, but more memory

    print('=== Runnning on device: p', device)

    if not cfg.DEBUG:
        wandb.init(
            name=cfg.NAME,
            project='continuous_urban_change_detection',
            tags=['run', 'urban', 'change detection', ],
        )

    try:
        run_training(cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
