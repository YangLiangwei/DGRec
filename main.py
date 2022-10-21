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
    sample_weight = dataloader.sample_weight.to(device)

    model = choose_model(args, dataloader)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    early_stop(99999.99, model)

    for epoch in range(args.epoch):
        model.train()

        loss_train = torch.zeros(1).to(device)

        graph_pos = dataloader.train_graph
        for i in range(args.neg_number):
            graph_neg = construct_negative_graph(graph_pos, ('user', 'rate', 'item'))

            score_pos, score_neg = model(graph_pos, graph_neg)
            if not args.category_balance:
                loss_train += -(score_pos - score_neg).sigmoid().log().mean()
            else:
                loss = -(score_pos - score_neg).sigmoid().log()
                items = graph_pos.edges(etype = 'rate')[1]
                weight = sample_weight[items]
                loss_train += (weight * loss.squeeze(1)).mean()

        loss_train = loss_train / args.neg_number
        logging.info('train loss = {}'.format(loss_train.item()))
        opt.zero_grad()
        loss_train.backward()
        opt.step()

        model.eval()
        graph_val_pos = dataloader.val_graph
        graph_val_neg = construct_negative_graph(graph_val_pos, ('user', 'rate', 'item'))

        score_pos, score_neg = model(graph_val_pos, graph_val_neg)
        if not args.category_balance:
            loss_val = -(score_pos - score_neg).sigmoid().log().mean()
        else:
            loss = -(score_pos - score_neg).sigmoid().log()
            items = graph_val_pos.edges(etype = 'rate')[1]
            weight = sample_weight[items]
            loss_val = (weight * loss.squeeze(1)).mean()


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

