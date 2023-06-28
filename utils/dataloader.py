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
        self.user_number = 0
        self.item_number = 0
        self.device = device

        self.train_graph, self.dataloader_train = self.read_train_graph(self.train_path)
        logging.info('reading positive train data')
        self.train_graph_positive, self.dataloader_train_positive = self.read_train_graph_positive(self.train_path)
        # logging.info('reading negative train data')
        # self.train_graph_negative, self.dataloader_train_negative = self.read_train_graph_negative(self.train_path)
        logging.info('reading valid data')
        self.val_graph, self.dataloader_val = self.read_val_graph(self.val_path)
        logging.info('reading test data')
        self.test_dic, self.dataloader_test = self.read_test(self.test_path)

        # read node dataloader for graphau
        # if self.args.model == 'graphau':
        #     logging.info('begin reading multi-hop graphs')
        #     self.dataloader_multi_hop_train = self.get_multi_hop_graphs(self.train_graph_positive)
        #     self.dataloader_multi_hop_val = self.get_multi_hop_graphs(self.val_graph)

    def get_csr_matrix(self, array):
        users = array[:, 0]
        items = array[:, 1]
        data = np.ones(len(users))
        # return torch.sparse_coo_tensor(array.t(), data, dtype = bool).to_sparse_csr().to(args.device)
        return coo_matrix((data, (users, items)), shape = (self.user_number, self.item_number), dtype = bool).tocsr()


    def get_multi_hop_graphs(self, hetergraph):
        users = hetergraph.edges(etype = ('user', 'rate', 'item'))[0]
        items = hetergraph.edges(etype = ('user', 'rate', 'item'))[1] + self.user_number
        src = torch.cat([users, items])
        dst = torch.cat([items, users])

        h = 1
        graph_homo = dgl.graph((src, dst), num_nodes = self.user_number + self.item_number)
        src = graph_homo.edges()[0]
        dst = graph_homo.edges()[1]
        hop = torch.tensor([h] * graph_homo.edges()[0].shape[0])

        for i in range(self.args.layers - 1):
            print(i)
            h += 1
            graph_hop = dgl.khop_graph(graph_homo, k = i + 2)
            src = torch.cat([src, graph_hop.edges()[0]])
            dst = torch.cat([dst, graph_hop.edges()[1]])
            hop = torch.cat([hop, torch.tensor([h] * graph_hop.edges()[0].shape[0])])

        data = torch.stack([src, dst, hop]).t()
        # data[data >= self.user_number] -= self.user_number
        print("total number of aligned edges")
        print(len(data))

        dataset = torch.utils.data.TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size = self.args.batch_size, shuffle = True, num_workers = 4)

        return dataloader



    def get_homo_graphs(self, hetergraph):
        users = hetergraph.edges(etype = ('user', 'rate', 'item'))[0]
        items = hetergraph.edges(etype = ('user', 'rate', 'item'))[1] + self.user_number
        src = torch.cat([users, items])
        dst = torch.cat([items, users])

        graph_homo_1 = dgl.graph((src, dst), num_nodes = self.user_number + self.item_number)
        graph_list = [graph_homo_1]

        for i in range(self.args.layers - 1):
            graph_list.append(dgl.khop_graph(graph_homo_1, k = i + 2))

        dataset = torch.utils.data.TensorDataset(torch.tensor(range(self.user_number + self.item_number)))
        dataloader_node = DataLoader(dataset, batch_size = self.args.batch_size, shuffle = True, num_workers = 4)

        return graph_list, dataloader_node

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
                rating = int(line[2])
                train_data.append([user, item, rating])

                if user in self.historical_dict:
                    self.historical_dict[user].add(item)
                else:
                    self.historical_dict[user] = set([item])

        train_data = torch.tensor(train_data)
        self.user_number = max(self.user_number, train_data[:, 0].max() + 1)
        self.item_number = max(self.item_number, train_data[:, 1].max() + 1)

        # train_data = self.stacking_layers(train_data, 1)

        graph_data = {
            ('user', 'rate', 'item'): (train_data[:, 0].long(), train_data[:, 1].long()),
            ('item', 'rated by', 'user'): (train_data[:, 1].long(), train_data[:, 0].long())
        }
        node_dict = {'user': self.user_number, 'item': self.item_number}
        graph = dgl.heterograph(graph_data, node_dict)
        dataset = torch.utils.data.TensorDataset(train_data)
        dataloader = DataLoader(dataset, batch_size = self.args.batch_size, shuffle = True, num_workers = 1)

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

        return graph, dataloader

    def read_train_graph_positive(self, path):
        self.historical_dict = {}
        train_data = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip().split(',')
                user = int(line[0])
                item = int(line[1])
                rating = int(line[2])
                if rating > 3:
                    train_data.append([user, item, rating])

                if user in self.historical_dict:
                    self.historical_dict[user].add(item)
                else:
                    self.historical_dict[user] = set([item])

        train_data = torch.tensor(train_data)
        self.train_csr = self.get_csr_matrix(train_data)

        # train_data = self.stacking_layers(train_data, 1)

        graph_data = {
            ('user', 'rate', 'item'): (train_data[:, 0].long(), train_data[:, 1].long()),
            ('item', 'rated by', 'user'): (train_data[:, 1].long(), train_data[:, 0].long())
        }
        node_dict = {'user': self.user_number, 'item': self.item_number}
        graph = dgl.heterograph(graph_data, node_dict)

        dataset = torch.utils.data.TensorDataset(train_data)
        dataloader = DataLoader(dataset, batch_size = self.args.batch_size, shuffle = True, num_workers = 1)

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

        return graph, dataloader


    def read_train_graph_negative(self, path):
        train_data = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip().split(',')
                user = int(line[0])
                item = int(line[1])
                rating = int(line[2])
                if rating < 3:
                    train_data.append([user, item])

        train_data = torch.tensor(train_data)

        # train_data = self.stacking_layers(train_data, 1)

        graph_data = {
            ('user', 'rate', 'item'): (train_data[:, 0].long(), train_data[:, 1].long()),
            ('item', 'rated by', 'user'): (train_data[:, 1].long(), train_data[:, 0].long())
        }
        node_dict = {'user': self.user_number, 'item': self.item_number}
        graph = dgl.heterograph(graph_data, node_dict)
        dataset = torch.utils.data.TensorDataset(train_data)
        dataloader = DataLoader(dataset, batch_size = self.args.batch_size, shuffle = True, num_workers = 1)

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

        return graph, dataloader

    def read_val_graph(self, path):
        val_data = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip().split(',')
                user = int(line[0])
                item = int(line[1])
                rating = int(line[2])

                if self.args.model in ['signedau']:
                    val_data.append([user, item, rating])
                else:
                    if rating > 3:
                        val_data.append([user, item, rating])

        val_data = torch.tensor(val_data)

        graph_data = {
            ('user', 'rate', 'item'): (val_data[:, 0].long(), val_data[:, 1].long()),
            ('item', 'rated by', 'user'): (val_data[:, 1].long(), val_data[:, 0].long())
        }
        number_nodes_dict = {'user': self.user_number, 'item': self.item_number}
        graph = dgl.heterograph(graph_data, num_nodes_dict = number_nodes_dict)

        dataset = torch.utils.data.TensorDataset(val_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.args.batch_size, shuffle = True)

        return graph, dataloader

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
