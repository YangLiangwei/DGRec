import torch.nn as nn
from tqdm import tqdm
import torch as th
import pdb
import torch.nn.functional as F
import torch
import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl.nn import GraphConv
from models.layers import LightGCNLayer, SubLightGCNLayer, GCNLayer

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
        self.graph = dataloader.train_graph
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

class DGRec(BaseGraphModel):
    def __init__(self, args, dataloader):
        super(DGRec, self).__init__(args, dataloader)
        self.W = torch.nn.Parameter(torch.randn(self.args.embed_size, self.args.embed_size))
        self.a = torch.nn.Parameter(torch.randn(self.args.embed_size))
        # self.a = torch.nn.Parameter(torch.ones(self.args.layers + 1) / (self.args.layers + 1))

    def build_layer(self, idx):
        return SubLightGCNLayer(self.args)

    def det_weight(self, ls):
        tensor = torch.stack(ls, dim = 0)

        dist = torch.cdist(tensor, tensor)
        weight = torch.det(dist, dim = 1).sum(-1)
        weight = F.softmax(weight).unsqueeze(1).unsqueeze(1)
        weight = weight.clone().detach()
        tensor = (tensor * weight).sum(0)
        return tensor


    def var_weight(self, ls):
        tensor = torch.stack(ls, dim = 0)
        tensor_max = torch.max(tensor, dim = 1)[0].unsqueeze(1)
        tensor_min = torch.min(tensor, dim = 1)[0].unsqueeze(1)
        normalized = (tensor - tensor_min) / (tensor_max - tensor_min)
        weight = torch.var(normalized, dim = 1).sum(-1)
        weight = F.softmax(weight).unsqueeze(1).unsqueeze(1)
        weight = weight.clone().detach()
        tensor = (tensor * weight).sum(0)
        return tensor

    def layer_attention(self, ls, W, a):
        tensor_layers = torch.stack(ls, dim = 0)
        weight = torch.matmul(tensor_layers, W)
        weight = F.softmax(torch.matmul(weight, a), dim = 0).unsqueeze(-1)
        tensor_layers = torch.sum(tensor_layers * weight, dim = 0)
        return tensor_layers

    def get_embedding(self):
        user_embed = [self.user_embedding]
        item_embed = [self.item_embedding]
        h = self.node_features

        for layer in self.layers:

            h_item = layer(self.graph, h, ('user', 'rate', 'item'))
            h_user = layer(self.graph, h, ('item', 'rated by', 'user'))
            h = {'user': h_user, 'item': h_item}
            user_embed.append(h_user)
            item_embed.append(h_item)
        # user_embed = torch.mean(torch.stack(user_embed, dim = 0), dim = 0)
        # item_embed = torch.mean(torch.stack(item_embed, dim = 0), dim = 0)
        user_embed = self.layer_attention(user_embed, self.W, self.a)
        item_embed = self.layer_attention(item_embed, self.W, self.a)
        # user_embed = self.var_weight(user_embed)
        # item_embed = self.var_weight(item_embed)
        h = {'user': user_embed, 'item': item_embed}
        return h

