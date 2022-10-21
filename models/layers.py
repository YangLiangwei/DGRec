import torch.nn as nn
import torch as th
import pdb
from tqdm import tqdm
import torch.nn.functional as F
import dgl.function as fn
import dgl.nn as dglnn
import numpy as np
from fast_pytorch_kmeans import KMeans


class GCNLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.k = args.k
        self.dim = args.embed_size
        self.weight = th.nn.Parameter(th.randn(self.dim, self.dim))
        self.bias = th.nn.Parameter(th.zeros(self.dim))
        th.nn.init.xavier_uniform_(self.weight)

    def reduction(self, nodes):
        # -1 indicate user-> node, which does not include category information
        mail = nodes.mailbox['m']
        batch_size, neighbor_size, feature_size = mail.shape

        if nodes.mailbox['m'].shape[1] <= self.k:
            mail = mail.sum(dim = 1)
        else:
            weight = th.ones(batch_size, neighbor_size, device = mail.device)
            selected = th.multinomial(weight, self.k)
            mail = mail[th.arange(batch_size, dtype = th.long, device = mail.device).unsqueeze(-1), selected]
            mail = mail.sum(dim = 1)
        return {'h': mail}


    def forward(self, graph, h, etype):
        with graph.local_scope():
            src, _ , dst = etype
            feat_src = h[src]
            feat_dst = h[dst]
            aggregate_fn = fn.copy_src('h', 'm')

            degs = graph.out_degrees(etype = etype).float().clamp(min = 1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = th.reshape(norm, shp)
            feat_src = feat_src * norm

            graph.nodes[src].data['h'] = feat_src
            graph.update_all(aggregate_fn, self.reduction, etype = etype)

            # rst = graph.dstdata['h'][dst]
            rst = graph.nodes[dst].data['h']
            rst = th.matmul(rst, self.weight)
            degs = graph.in_degrees(etype = etype).float().clamp(min = 1)
            degs = th.clamp(degs, 0, self.k)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = th.reshape(norm, shp)
            rst = rst * norm
            rst += self.bias
            return rst


class LightGCNLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, h, etype):
        with graph.local_scope():
            src, _ , dst = etype
            feat_src = h[src]
            feat_dst = h[dst]
            aggregate_fn = fn.copy_src('h', 'm')

            degs = graph.out_degrees(etype = etype).float().clamp(min = 1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = th.reshape(norm, shp)
            feat_src = feat_src * norm

            graph.nodes[src].data['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg = 'm', out = 'h'), etype = etype)

            # rst = graph.dstdata['h'][dst]
            rst = graph.nodes[dst].data['h']
            degs = graph.in_degrees(etype = etype).float().clamp(min = 1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = th.reshape(norm, shp)
            rst = rst * norm
            return rst

class SubLightGCNLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.k = args.k
        self.sigma = args.sigma
        self.gamma = args.gamma
        self.cluster_num = args.cluster_num

    def similarity_matrix(self, X, sigma = 1.0, gamma = 2.0):
        dists = th.cdist(X, X)
        sims = th.exp(-dists / (sigma * dists.mean(dim = -1).mean(dim = -1).reshape(-1, 1, 1)))
        return sims

    def submodular_selection_feature(self, nodes):
        device = nodes.mailbox['m'].device
        feature = nodes.mailbox['m']
        sims = self.similarity_matrix(feature, self.sigma, self.gamma)

        batch_num, neighbor_num, feature_size = feature.shape
        nodes_selected = []
        cache = th.zeros((batch_num, 1, neighbor_num), device = device)
        # mask = th.ones(batch_num, neighbor_num, dtype = th.bool, device = device)

        for i in range(self.k):
            # gain = th.sum(th.maximum(th.masked_select(sims, mask.unsqueeze(-1)).reshape(batch_num, -1, neighbor_num), cache) - cache, dim = -1)
            gain = th.sum(th.maximum(sims, cache) - cache, dim = -1)

            selected = th.argmax(gain, dim = 1)
            cache = th.maximum(sims[th.arange(batch_num, device = device), selected].unsqueeze(1), cache)
            # mask[th.arange(batch_num, device = device), selected] = False

            nodes_selected.append(selected)

        return th.stack(nodes_selected).t()


    def submodular_selection_cluster(self, nodes):
        device = nodes.mailbox['c'].device
        cluster = nodes.mailbox['cluster']

        batch_num, neighbor_num, cluster_num = cluster.shape
        nodes_selected = []

        for i in range(self.k):
            gain = cluster.sum(-1)
            selected = th.argmax(gain, dim = 1)
            nodes_selected.append(selected)
            gain[th.arange(batch_num, dtype = th.long, device = device), selected] = -float('inf')
            mask = cluster[th.arange(batch_num, device = device), selected].unsqueeze(-1).repeat_interleave(neighbor_num, dim = -1).transpose(1, 2)
            mask = mask.bool()
            cluster[mask] -= 1
        return th.stack(nodes_selected).t()


    def submodular_selection_category(self, nodes):
        device = nodes.mailbox['c'].device
        category = nodes.mailbox['c'].squeeze(-1)
        feature = nodes.mailbox['m']

        batch_num, neighbor_num, _ = feature.shape
        nodes_selected = []
        gain = th.zeros((batch_num, neighbor_num), device = device)

        for i in range(self.k):
            selected = th.argmax(gain, dim = 1)
            nodes_selected.append(selected)
            gain[th.arange(batch_num, dtype = th.long, device = device), selected] = -float('inf')

            category_selected = category[th.arange(batch_num, dtype = th.long, device = device), selected]
            mask = category == category_selected.unsqueeze(-1)
            gain[mask] -= 1
        return th.stack(nodes_selected).t()


    def sub_reduction(self, nodes):
        # -1 indicate user-> node, which does not include category information
        mail = nodes.mailbox['m']
        batch_size, neighbor_size, feature_size = mail.shape

        if (-1 in nodes.mailbox['c']) or nodes.mailbox['m'].shape[1] <= self.k:
            mail = mail.sum(dim = 1)
        else:
            neighbors = self.submodular_selection_feature(nodes)
            # neighbors = self.submodular_selection_category(nodes)
            # neighbors = self.submodular_selection_cluster(nodes)
            mail = mail[th.arange(batch_size, dtype = th.long, device = mail.device).unsqueeze(-1), neighbors]
            mail = mail.sum(dim = 1)
        return {'h': mail}

    def category_aggregation(self, edges):
        # return {'c': edges.src['category'], 'm': edges.src['h'], 'cluster': edges.src['cluster']}
        return {'c': edges.src['category'], 'm': edges.src['h']}

    def get_cluster(self, feat):
        device = feat.device
        # feat = feat.cpu().detach().numpy()
        kmeans = KMeans(n_clusters = self.cluster_num, mode = 'euclidean')
        ls = []
        for i in range(feat.shape[1]):
            feature = feat[:, i].reshape(-1, 1)
            res = kmeans.fit_predict(feature)
            eye = th.eye(self.cluster_num, device = device)
            ls.append(eye[res])
        return th.cat(ls, dim = 1)

    def forward(self, graph, h, etype):
        with graph.local_scope():
            src, _ , dst = etype
            feat_src = h[src]
            feat_dst = h[dst]

            # if src == 'item':
            #     graph.ndata['cluster'] = {src: self.get_cluster(feat_src), dst: th.zeros(feat_dst.shape[0], 1, device = feat_dst.device)}
            # else:
            #     graph.ndata['cluster'] = {src: th.zeros(feat_src.shape[0], 1, device = feat_src.device), dst: th.zeros(feat_dst.shape[0], 1, device = feat_dst.device)}


            degs = graph.out_degrees(etype = etype).float().clamp(min = 1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = th.reshape(norm, shp)
            feat_src = feat_src * norm

            graph.nodes[src].data['h'] = feat_src
            graph.update_all(self.category_aggregation, self.sub_reduction, etype = etype)
            # graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'), etype = etype)

            # rst = graph.dstdata['h'][dst]
            rst = graph.nodes[dst].data['h']
            degs = graph.in_degrees(etype = etype).float().clamp(min = 1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = th.reshape(norm, shp)
            rst = rst * norm
            return rst
