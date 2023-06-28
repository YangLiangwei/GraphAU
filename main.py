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

    if args.model not in ['random', 'popularity', 'directau_mf', 'directau_lightgcn', 'signedau', 'graphau', 'multiloss']:
        for epoch in range(args.epoch):
            model.train()

            loss_train = torch.zeros(1).to(device)

            graph_pos_positive = dataloader.train_graph_positive.to(device)
            # graph_pos_negative = dataloader.train_graph_negative

            for i in range(args.neg_number):

                graph_neg = construct_negative_graph(graph_pos_positive, ('user', 'rate', 'item')).to(device)
                if args.model == 'ultragcn':
                    loss_train += model.compute_loss(graph_pos_positive, graph_neg)
                else:
                    score_pos_positive, score_neg_positive = model(graph_pos_positive, graph_neg)
                    loss_train += -(score_pos_positive - score_neg_positive).sigmoid().log().mean()

                # if args.model not in ['sbgnn']:
                #     graph_neg = construct_negative_graph(graph_pos_positive, ('user', 'rate', 'item'))
                #     score_pos_positive, score_neg_positive = model(graph_pos_positive, graph_neg)
                #     loss_train += -(score_pos_positive - score_neg_positive).sigmoid().log().mean()
                # else:
                #     graph_neg_positive = construct_negative_graph(graph_pos_positive, ('user', 'rate', 'item'))
                #     graph_neg_negative = construct_negative_graph(graph_pos_negative, ('user', 'rate', 'item'))
                #     score_pos_positive, score_neg_positive = model(graph_pos_positive, graph_neg_positive)
                #     score_pos_negative, score_neg_negative = model(graph_pos_negative, graph_neg_negative)

                    # loss_train += -(score_pos_positive - score_neg_positive).sigmoid().log().mean()
                    # loss_train += -(score_neg_negative - score_pos_negative).sigmoid().log().mean()

            loss_train = loss_train / args.neg_number
            logging.info('train loss = {}'.format(loss_train.item()))
            opt.zero_grad()
            loss_train.backward()
            opt.step()

            model.eval()
            graph_val_pos = dataloader.val_graph.to(device)
            graph_val_neg = construct_negative_graph(graph_val_pos, ('user', 'rate', 'item'))

            if args.model == 'ultragcn':
                loss_val  = model.compute_loss(graph_val_pos, graph_val_neg)
            else:
                score_pos, score_neg = model(graph_val_pos, graph_val_neg)
                loss_val = -(score_pos - score_neg).sigmoid().log().mean()

            early_stop(loss_val, model)

            if torch.isnan(loss_val) == True:
                break

            if early_stop.early_stop:
                break

    if args.model in ['graphau', 'directau_mf', 'directau_lightgcn', 'signedau', 'multiloss']:
        for epoch in range(args.epoch):
            model.train()

            loss_train_epoch = torch.zeros(1).to(device)
            count = 0

            if args.model in ['graphau', 'directau_mf', 'directau_lightgcn', 'multiloss']:
                dl = dataloader.dataloader_train_positive
                # dl = dataloader.dataloader_multi_hop_train
            if args.model == 'signedau':
                dl = dataloader.dataloader_train

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

    # if args.model in ['graphau']:
    #     for epoch in range(args.epoch):
    #         model.train()

    #         loss_train_epoch = torch.zeros(1).to(device)
    #         count = 0

    #         dl = dataloader.dataloader_multi_hop_train

    #         for data in tqdm(dl):
    #             data = data[0]
    #             count += 1

    #             users = data[:, 0].to(device)
    #             items = data[:, 1].to(device)
    #             hops = data[:, 2].to(device)
    #             loss_train = model.calculate_loss(users, items, hops)
    #             loss_train_epoch += loss_train.item()
    #             opt.zero_grad()
    #             loss_train.backward()
    #             opt.step()

    #         logging.info('train loss = {}'.format(loss_train_epoch.item() / count))

    #         if epoch > args.min_epoch:
    #             model.eval()

    #             loss_val_epoch = torch.zeros(1).to(device)
    #             count = 0
    #             for data in tqdm(dataloader.dataloader_multi_hop_val):
    #                 data = data[0]
    #                 count += 1

    #                 users = data[:, 0]
    #                 items = data[:, 1]
    #                 hops = data[:, 2]
    #                 loss_val = model.calculate_loss(users, items, hops)
    #                 loss_val_epoch += loss_val.item()

    #             loss_val = loss_val_epoch / count

    #             early_stop(loss_val, model)

    #             if torch.isnan(loss_val) == True:
    #                 break

    #             if early_stop.early_stop:
    #                 break


    # if args.model in ['graphau']:
    #     for epoch in range(args.epoch):
    #         model.train()

    #         loss_train_epoch = torch.zeros(1).to(device)
    #         count = 0

    #         dl = dataloader.dataloader_node

    #         for data in tqdm(dl):
    #             nodes = data[0]
    #             node_number = nodes.shape[0]
    #             count += node_number

    #             loss_train = model.calculate_loss(nodes, dataloader.homo_graph_list_train)
    #             loss_train_epoch += loss_train.item() * node_number
    #             opt.zero_grad()
    #             loss_train.backward()
    #             opt.step()

    #         logging.info('train loss = {}'.format(loss_train_epoch.item() / count))

    #         if epoch > args.min_epoch:
    #             model.eval()

    #             loss_val_epoch = torch.zeros(1).to(device)
    #             count = 0
    #             for data in tqdm(dl):
    #                 nodes = data[0]
    #                 node_number = nodes.shape[0]
    #                 count += node_number

    #                 loss_val = model.calculate_loss(nodes, dataloader.homo_graph_list_val)
    #                 loss_val_epoch += loss_val.item() * node_number

    #             loss_val = loss_val_epoch / count

    #             early_stop(loss_val, model)

    #             if torch.isnan(loss_val) == True:
    #                 break

    #             if early_stop.early_stop:
    #                 break

    logging.info('loading best model for test')
    model.load_state_dict(torch.load(early_stop.save_path))
    # args.model_mf = load_mf_model(args, dataloader)
    tester = Tester(args, model, dataloader)
    logging.info('begin testing')
    res = tester.test()

