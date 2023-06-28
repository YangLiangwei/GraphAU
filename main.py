import sys
import dgl
import dgl.function as fn
import os
import multiprocessing as mp
from tqdm import tqdm
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from utils.parser import parse_args
from utils.dataloader import Dataloader
from utils.utils import config, construct_negative_graph, choose_model, load_mf_model, NegativeGraph
from utils.tester import Tester
from models.sampler import NegativeSampler

if __name__ == '__main__':
    args = parse_args()
    early_stop = config(args)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    device = torch.device(device)
    args.device = device

    data = args.dataset
    dataloader = Dataloader(args, data, device)
    # NegativeGraphConstructor = NegativeGraph(dataloader.historical_dict)

    model = choose_model(args, dataloader)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    early_stop(999999999999.99, model)

    for epoch in range(args.epoch):
        model.train()

        loss_train_epoch = torch.zeros(1).to(device)
        count = 0

        dl = dataloader.dataloader_train_positive

        for data in tqdm(dl):
            data = data[0]
            count += 1

            users = data[:, 0]
            items = data[:, 1]
            ratings = data[:, 2]
            loss_train = model.calculate_loss(users, items, ratings)
            loss_train_epoch += loss_train.item()
            opt.zero_grad()
            loss_train.backward()
            opt.step()

        logging.info('train loss = {}'.format(loss_train_epoch.item() / count))

        if epoch > args.min_epoch:
            model.eval()

            loss_val_epoch = torch.zeros(1).to(device)
            count = 0
            for data in tqdm(dataloader.dataloader_val):
                data = data[0]
                count += 1

                users = data[:, 0]
                items = data[:, 1]
                ratings = data[:, 2]
                loss_val = model.calculate_loss(users, items, ratings)
                loss_val_epoch += loss_val.item()

            loss_val = loss_val_epoch / count

            early_stop(loss_val, model)

            if torch.isnan(loss_val) == True:
                break

            if early_stop.early_stop:
                break

    logging.info('loading best model for test')
    model.load_state_dict(torch.load(early_stop.save_path))
    # args.model_mf = load_mf_model(args, dataloader)
    tester = Tester(args, model, dataloader)
    logging.info('begin testing')
    res = tester.test()

