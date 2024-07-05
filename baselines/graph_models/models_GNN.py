#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Understanding Survey Paper Taxonomy about Large Language Models via Graph Representation Learning
@topic: GNN Models.
@author: Jun Zhuang, Casey Kennington
"""

from dgl.nn.pytorch import GraphConv, SAGEConv, GINConv, TAGConv
import torch


# GNNs model --------------------
class GNNs(torch.nn.Module):
    def __init__(self,
                 graph,
                 in_feats,
                 n_hidden,
                 n_classes,
                 activation,
                 dropout,
                 model_name,  # graph operator
                 **kwargs):
        super(GNNs, self).__init__()
        self.graph = graph  # graph DGLGraph
        self.aggregator_type = kwargs["aggregator_type"]
        self.n_filter = kwargs["n_filter"]
        self.dropout = dropout
        # Select the model layers
        if model_name == "GCN":
            model_in = GraphConv(in_feats, n_hidden, activation=activation, allow_zero_in_degree=True)
            model_h = GraphConv(n_hidden, n_hidden, activation=activation, allow_zero_in_degree=True)
            model_out = GraphConv(n_hidden, n_classes, allow_zero_in_degree=True)
            self.w0, self.w1 = model_in.weight, model_out.weight
        # the following three models are not validated in the paper.
        elif model_name == "GraphSAGE":  # Aggregator type: mean, gcn, pool, lstm.
            model_in = SAGEConv(in_feats, n_hidden, self.aggregator_type, activation=activation)
            model_h = SAGEConv(n_hidden, n_hidden, self.aggregator_type, activation=activation)
            model_out = SAGEConv(n_hidden, n_classes, self.aggregator_type)
            self.w0, self.w1 = \
                torch.transpose(model_in.fc_neigh.weight, 0, 1), torch.transpose(model_out.fc_neigh.weight, 0, 1)
        elif model_name == "GIN": # Aggregator type: sum, max or mean.
            model_in = GINConv(torch.nn.Linear(in_feats, n_hidden), self.aggregator_type, init_eps=0)
            model_h = GINConv(torch.nn.Linear(n_hidden, n_hidden), self.aggregator_type, init_eps=0)
            model_out = GINConv(torch.nn.Linear(n_hidden, n_classes), self.aggregator_type, init_eps=0)
        elif model_name == "TAGCN": # k for the size of filter
            model_in = TAGConv(in_feats, n_hidden, k=self.n_filter, activation=activation)
            model_h = TAGConv(n_hidden, n_hidden, k=self.n_filter, activation=activation)
            model_out = TAGConv(n_hidden, n_classes, k=self.n_filter)
        else:
            print("model_name is incorrect!")
            return 0
        # Build the models
        model_do = torch.nn.Dropout(p=self.dropout)
        model_layers = [model_in, model_do, model_h, model_do, model_out]
        self.layers = torch.nn.ModuleList(model_layers)


    def forward(self, feat):
        # Input layer
        layer_in = self.layers[0](self.graph, feat)
        layer_in = self.layers[1](layer_in)  # dropout
        # Hidden layer
        self.layer_h = self.layers[2](self.graph, layer_in)
        # Output layer
        layer_out = self.layers[3](self.layer_h)  # dropout
        layer_out = self.layers[4](self.graph, layer_out)
        return layer_out, self.layer_h
