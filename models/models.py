import torch.nn as nn
from tqdm import tqdm
import torch as th
import pdb
import torch.nn.functional as F
import torch
import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl.nn import GraphConv, GATConv
import numpy as np
import math
from models.layers import LightGCNLayer, SubLightGCNLayer, GCNLayer, FilterGNNLayer, LinearTransformLayer, SBGNNUpdate, NGCFLayer

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype = etype)
            return graph.edges[etype].data['score']

class BaseGraphModel(nn.Module):
    def __init__(self, args, dataloader):
        super().__init__()
        self.args = args
        self.hid_dim = args.embed_size
        self.layer_num = args.layers
        self.graph = dataloader.train_graph_positive.to(args.device)
        self.user_number = dataloader.user_number
        self.item_number = dataloader.item_number


        self.user_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('user').shape[0], self.hid_dim))
        self.item_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('item').shape[0], self.hid_dim))
        self.predictor = HeteroDotProductPredictor()

        self.build_model()

        self.node_features = {'user': self.user_embedding, 'item': self.item_embedding}

    def build_layer(self, idx):
        pass

    def build_model(self):
        self.layers = nn.ModuleList()
        for idx in range(self.layer_num):
            h2h = self.build_layer(idx)
            self.layers.append(h2h)

    def get_embedding(self):
        h = self.node_features

        graph_user2item = dgl.edge_type_subgraph(self.graph, ['rate'])
        graph_item2user = dgl.edge_type_subgraph(self.graph, ['rated by'])

        for layer in self.layers:
            user_feat = h['user']
            item_feat = h['item']

            h_item = layer(graph_user2item, (user_feat, item_feat))
            h_user = layer(graph_item2user, (item_feat, user_feat))

            h = {'user': h_user, 'item': h_item}
        return h

    def forward(self, graph_pos, graph_neg):
        h = self.get_embedding()
        score_pos = self.predictor(graph_pos, h, 'rate')
        score_neg = self.predictor(graph_neg, h, 'rate')
        return score_pos, score_neg

    def get_score(self, h, users):
        user_embed = h['user'][users]
        item_embed = h['item']
        scores = torch.mm(user_embed, item_embed.t())
        return scores

class GCNModel(BaseGraphModel):
    def __init__(self, args, dataloader):
        super(GCNModel, self).__init__(args, dataloader)

    def build_layer(self, idx):
        return GraphConv(self.hid_dim, self.hid_dim, norm = 'both', weight = True, bias = True, allow_zero_in_degree = True)


class GATModel(BaseGraphModel):
    def __init__(self, args, dataloader):
        super(GATModel, self).__init__(args, dataloader)

    def build_layer(self, idx):
        return GATConv(self.hid_dim, self.hid_dim, self.args.head, allow_zero_in_degree = True)

    def get_embedding(self):
        h = self.node_features

        graph_user2item = dgl.edge_type_subgraph(self.graph, ['rate'])
        graph_item2user = dgl.edge_type_subgraph(self.graph, ['rated by'])

        for layer in self.layers:
            user_feat = h['user']
            item_feat = h['item']

            h_item = layer(graph_user2item, (user_feat, item_feat))
            h_user = layer(graph_item2user, (item_feat, user_feat))
            h_item = h_item.mean(1)
            h_user = h_user.mean(1)

            h = {'user': h_user, 'item': h_item}
        return h

class NGCF(BaseGraphModel):
    def __init__(self, args, dataloader):
        super(NGCF, self).__init__(args, dataloader)
        self.args = args

    def build_layer(self, idx):
        return NGCFLayer(self.args)

    def get_embedding(self):
        user_embed = [self.user_embedding]
        item_embed = [self.item_embedding]
        h = self.node_features

        for layer in self.layers:

            h_item = layer(self.graph, h, ('user', 'rate', 'item'))
            h_user = layer(self.graph, h, ('item', 'rated by', 'user'))

            user_embed.append(h_user)
            item_embed.append(h_item)
            h = {'user': h_user, 'item': h_item}

        user_embed = torch.cat(user_embed, 1)
        item_embed = torch.cat(item_embed, 1)
        h = {'user': user_embed, 'item': item_embed}

        return h


class LightGCN(BaseGraphModel):
    def __init__(self, args, dataloader):
        super(LightGCN, self).__init__(args, dataloader)

    def build_layer(self, idx):
        return LightGCNLayer()

    def get_embedding(self):
        user_embed = [self.user_embedding]
        item_embed = [self.item_embedding]
        h = self.node_features

        for layer in self.layers:

            h_item = layer(self.graph, h, ('user', 'rate', 'item'))
            h_user = layer(self.graph, h, ('item', 'rated by', 'user'))

            user_embed.append(h_user)
            item_embed.append(h_item)
            h = {'user': h_user, 'item': h_item}

        user_embed = torch.mean(torch.stack(user_embed, dim = 0), dim = 0)
        item_embed = torch.mean(torch.stack(item_embed, dim = 0), dim = 0)
        h = {'user': user_embed, 'item': item_embed}

        return h

class Popularity(BaseGraphModel):
    def __init__(self, args, dataloader):
        super(Popularity, self).__init__(args, dataloader)
        self.csr = dataloader.train_csr
        self.popularity = torch.tensor(self.csr.toarray().sum(axis = 0))

    def build_model(self):
        pass

    def get_embedding(self):
        return None

    def get_score(self, h, users):
        return self.popularity.repeat(len(users), 1).float()

class MF(BaseGraphModel):
    def __init__(self, args, dataloader):
        super(MF, self).__init__(args, dataloader)

    def build_model(self):
        pass

    def get_embedding(self):
        return self.node_features

    def get_score(self, h, users):
        user_embedding = h['user'][users]
        item_embedding = h['item']
        return torch.mm(user_embedding, item_embedding.t())

    def get_embedding(self):
        return self.node_features

class DirectAU_MF(MF):
    def __init__(self, args, dataloader):
        super(DirectAU_MF, self).__init__(args, dataloader)
        self.gamma = args.gamma_au

    def get_embedding(self):
        return self.node_features

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_loss(self, users, items, ratings):
        mask = ratings > 3
        h = self.get_embedding()
        user_e = h['user'][users[mask]]
        item_e = h['item'][items[mask]]

        align = self.alignment(user_e, item_e)
        uniform = (self.uniformity(user_e) + self.uniformity(item_e)) / 2

        loss = align + self.gamma * uniform
        return loss

class GraphAU(MF):
    def __init__(self, args, dataloader):
        super(GraphAU, self).__init__(args, dataloader)
        self.gamma = args.gamma_au
        self.aggregation = LightGCNLayer()
        self.layers = args.layers
        self.graph = dataloader.train_graph_positive.to(args.device)
        self.decay_base = args.decaying_base
        ls = [1.0]
        for l in range(self.layers - 1):
            ls.append(ls[-1] * self.decay_base)
        self.decay_weight = th.tensor(ls).to(args.device)

    def get_embedding(self):
        return self.node_features

    def get_embedding_aggregation(self, hops):
        h = self.node_features

        for i in range(hops - 1):

            h_item = self.aggregation(self.graph, h, ('user', 'rate', 'item'))
            h_user = self.aggregation(self.graph, h, ('item', 'rated by', 'user'))

            h = {'user': h_user, 'item': h_item}
        return h

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_loss(self, users, items, ratings):
        mask = ratings > 3
        h = self.get_embedding()
        user_e = h['user'][users[mask]]
        item_e = h['item'][items[mask]]

        align = [self.alignment(user_e, item_e)]
        uniform = [(self.uniformity(user_e) + self.uniformity(item_e)) / 2]

        for i in range(2, self.layers + 1):
            h_agg = self.get_embedding_aggregation(i)
            user_e_agg = h_agg['user'][users[mask]]
            item_e_agg = h_agg['item'][items[mask]]
            align.append((self.alignment(user_e, item_e_agg) + self.alignment(user_e_agg, item_e)) / 2)
        align = torch.mean(self.decay_weight * torch.stack(align))
        # align = align[-1]

        loss = align + self.gamma * uniform[0]
        return loss

