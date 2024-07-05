#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Understanding Survey Paper Taxonomy about Large Language Models via Graph Representation Learning
@topic: Load dataset.
@author: Jun Zhuang, Casey Kennington
"""

import dgl.data
import torch
from utils import preprocess_text_graph


class LoadDataset():
    def __init__(self, data_name):
        self.data_name = data_name

    def load_text_graph_data(self, graph_path, label_path):
        """Load text graph dataset based on given data_name and pathes."""
        #print("Current dataset: mr, ohsumed, R52, R8, 20ng.")
        print("Selecting {0} Dataset ...".format(self.data_name))
        # Preprocess the text graph
        adj_sp, feat, labels_pad, train_lst, test_lst = \
                preprocess_text_graph(self.data_name, graph_path, label_path)
        # Convert to a dgl graph
        graph_dgl = dgl.from_scipy(adj_sp)
        graph_dgl.ndata['feat'] = feat  # torch.sparse_coo
        graph_dgl.ndata['label'] = torch.LongTensor(labels_pad)  # int64
        print("{0} Dataset Loaded!".format(self.data_name))
        # Update the train/test id (list)
        self.train_lst, self.test_lst = train_lst, test_lst
        return graph_dgl
