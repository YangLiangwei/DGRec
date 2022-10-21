import sys
from tqdm import tqdm
import pdb
import torch
import logging
import numpy as np
import dgl
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import coo_matrix, csr_matrix

class TestDataset(Dataset):
    def __init__(self, dic):
        self.keys = torch.tensor(list(dic.keys()), dtype = torch.long)
        ls_values = [tensor for tensor in dic.values()]
        self.values = csr_matrix(torch.stack(ls_values))

    def __getitem__(self, index):
        key = self.keys[index]
        values = self.values[index]
        return {'key': key, 'value': values}

    def __len__(self):
        return len(self.keys)

class Dataloader(object):
    def __init__(self, args, data, device):
        logging.info("loadding data")
        self.args = args
        self.train_path = './datasets/' + data + '/train.txt'
        self.val_path = './datasets/' + data + '/val.txt'
        self.test_path = './datasets/' + data + '/test.txt'
        self.category_path = './datasets/' + data + '/item_category.txt'
        self.user_number = 0
        self.item_number = 0
        self.device = device
        logging.info('reading category information')
        self.category_dic, self.category_num = self.read_category(self.category_path)
        logging.info('reading train data')
        self.train_graph, self.dataloader_train = self.read_train_graph(self.train_path)
        logging.info('reading valid data')
        self.val_graph, self.dataloader_val = self.read_val_graph(self.val_path)
        logging.info('reading test data')
        self.test_dic, self.dataloader_test = self.read_test(self.test_path)
        logging.info('get weight for each sample')
        self.sample_weight = self.get_sample_weight(self.category_dic)

    def get_csr_matrix(self, array):
        users = array[:, 0]
        items = array[:, 1]
        data = np.ones(len(users))
        # return torch.sparse_coo_tensor(array.t(), data, dtype = bool).to_sparse_csr().to(args.device)
        return coo_matrix((data, (users, items)), shape = (self.user_number, self.item_number), dtype = bool).tocsr()

    def read_category(self, path):
        num = 0
        dic = {}
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip().split(',')
                item = int(line[0])
                category = int(line[1])
                if category not in dic.values():
                    num += 1
                dic[item] = category
        return dic, num

    def get_sample_weight(self, category_dic):
        dic = {}
        categories = category_dic.values()
        for c in tqdm(categories):
            if c in dic:
                dic[c] += 1
            else:
                dic[c] = 1
        weight_tensor = torch.tensor(list(dic.values()), dtype = torch.float)
        effective_num = 1.0 - torch.pow(self.args.beta_class, weight_tensor)
        weight = (1 - self.args.beta_class) / effective_num
        weight = weight / weight.sum() * self.category_num

        return weight[torch.tensor(list(category_dic.values()))]
        

    def stacking_layers(self, array, num):
        pdb.set_trace()
        count, _ = array.shape
        data = np.ones(count)

        user2item =  torch.sparse_coo_tensor(array.t(), data).to(self.args.device)
        item2user = user2item.t()
        trans = torch.sparse.mm(item2user, user2item)

        res = user2item
        for i in range(num):
            res = torch.sparse.mm(res, trans)

        return array

    def read_train_graph(self, path):
        self.historical_dict = {}
        train_data = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip().split(',')
                user = int(line[0])
                item = int(line[1])
                train_data.append([user, item])

                if user in self.historical_dict:
                    self.historical_dict[user].add(item)
                else:
                    self.historical_dict[user] = set([item])

        train_data = torch.tensor(train_data)
        self.user_number = max(self.user_number, train_data[:, 0].max() + 1)
        self.item_number = max(self.item_number, train_data[:, 1].max() + 1)
        self.train_csr = self.get_csr_matrix(train_data)

        # train_data = self.stacking_layers(train_data, 1)

        graph_data = {
            ('user', 'rate', 'item'): (train_data[:, 0].long(), train_data[:, 1].long()),
            ('item', 'rated by', 'user'): (train_data[:, 1].long(), train_data[:, 0].long())
        }
        graph = dgl.heterograph(graph_data)
        category_tensor = torch.tensor(list(self.category_dic.values()), dtype = torch.long).unsqueeze(1)
        graph.ndata['category'] = {'item': category_tensor, 'user': torch.zeros(self.user_number, 1) - 1}
        dataset = torch.utils.data.TensorDataset(train_data)
        dataloader = DataLoader(dataset, batch_size = self.args.batch_size, shuffle = True, num_workers = 4)

        # train_eid_dict = {('user', 'rate', 'item'): torch.arange(train_data.shape[0])}
        # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.args.layers)
        # dataloader = dgl.dataloading.EdgeDataLoader(
        #     graph, train_eid_dict, sampler,
        #     negative_sampler = dgl.dataloading.negative_sampler.Uniform(4),
        #     batch_size = self.args.batch_size,
        #     shuffle = True,
        #     drop_last = False,
        #     num_workers = 4
        # )

        return graph.to(self.device), dataloader

    def read_val_graph(self, path):
        val_data = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip().split(',')
                user = int(line[0])
                item = int(line[1])
                val_data.append([user, item])

        val_data = torch.tensor(val_data)

        graph_data = {
            ('user', 'rate', 'item'): (val_data[:, 0].long(), val_data[:, 1].long()),
            ('item', 'rated by', 'user'): (val_data[:, 1].long(), val_data[:, 0].long())
        }
        number_nodes_dict = {'user': self.user_number, 'item': self.item_number}
        graph = dgl.heterograph(graph_data, num_nodes_dict = number_nodes_dict)

        dataset = torch.utils.data.TensorDataset(val_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.args.batch_size, shuffle = True)

        return graph.to(self.device), dataloader
    
    def read_test(self, path):
        dic_test = {}
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                user = int(line[0])
                item = int(line[1])
                if user in dic_test:
                    dic_test[user].append(item)
                else:
                    dic_test[user] = [item]
        
        dataset = torch.utils.data.TensorDataset(torch.tensor(list(dic_test.keys()), dtype = torch.long, device = self.device))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.args.batch_size, shuffle = False)
        return dic_test, dataloader