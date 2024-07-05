#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Understanding Survey Paper Taxonomy about Large Language Models via Graph Representation Learning
@topic: Utils modules.
@author: Jun Zhuang, Casey Kennington
"""

import os
import sys
import pickle
import yaml
import torch
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as ss
from scipy.stats import entropy
from sklearn.metrics import f1_score
from zipfile import ZipFile


# ----- for text graphs -----
def read_file(path, mode='r', encoding=None):
    if mode not in {"r", "rb"}:
        raise ValueError("only read")
    return open(path, mode=mode, encoding=encoding)

def preprocess_adj(adj, is_sparse=False, is_torch=True):
    """Preprocessing of adjacency matrix for simple pygGCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + ss.eye(adj.shape[0]))
    if is_sparse:
        if is_torch:  # return pytorch sparse tensor
            adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
            return adj_normalized
        else:  # return scipy sparse coo_matrix
            return adj_normalized.tocoo().astype(np.float32)
    else:
        return torch.from_numpy(adj_normalized.A).float()

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = ss.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = ss.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_text_graph(data_name, graph_path, label_path):
    """Preprocess the text graph dataset."""
    # ----- Adjacency Matrix -----
    graph_filename = f"{graph_path}/{data_name}.txt"
    graph = nx.read_weighted_edgelist(graph_filename, nodetype=int)
    in_feats = graph.number_of_nodes()
    adj = nx.to_scipy_sparse_array(graph,
                                    nodelist=list(range(in_feats)),
                                    weight='weight',
                                    dtype=np.float32)
    #adj = nx.to_scipy_sparse_matrix(graph,
    #                                nodelist=list(range(in_feats)),
    #                                weight='weight',
    #                                dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_sp = preprocess_adj(adj, is_sparse=True, is_torch=False)  # sp.coo_matrix
    #self.adj = preprocess_adj(adj, is_sparse=True, is_torch=True)  # torch.sparse_coo
    # ----- Feature Matrix -----
    row = col = list(range(in_feats))
    indices = torch.from_numpy(np.vstack((row, col)).astype(np.int64))
    values = torch.FloatTensor([1.]*in_feats)
    shape = torch.Size((in_feats, in_feats))
    feat = torch.sparse.FloatTensor(indices, values, shape)  # torch.sparse_coo
    # ----- Labels -----
    label_filename = f"{label_path}/{data_name}.txt"
    labels = np.array(pd.read_csv(label_filename, sep="\t", header=None)[2])
    l2id = {label: indx for indx, label in enumerate(set(labels))}
    labels = [l2id[label] for label in labels]  # list
    # Pad the labels to the same number of nodes: unlabeled-node value = max_num_class
    labels_pad = labels + [len(l2id)]*(graph.number_of_nodes()-len(labels))
    # ----- Train & Test Split -----
    train_lst, test_lst = get_train_test(label_filename)  # list of node_id
    return adj_sp, feat, labels_pad, train_lst, test_lst

def get_train_test(label_filename):
    """Split the labeled nodes into train & test part."""
    train_lst, test_lst = list(), list()
    with read_file(label_filename, mode="r") as file:
        for indx, item in enumerate(file):
            if item.split("\t")[1] in {"train", "training", "20news-bydate-train"}:
                train_lst.append(indx)
            else:
                test_lst.append(indx)
    return train_lst, test_lst

def build_bool_mask(labels, node_id_lst):
    """ Build boolean mask based on the given node ids. """
    mask = torch.zeros([labels.shape[0]], dtype=torch.bool)
    mask[node_id_lst] = True
    return mask


# ----- main -----
def check_mkdirs(dir_name):
    """Create a data path if necessary."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

def read_pickle(file_name):
    """Load the dataset"""
    with open (file_name,'rb') as file:
        return pickle.load(file)

def dump_pickle(file_name, data):
    """Export the dataset"""
    with open (file_name,'wb') as file:
        pickle.dump(data, file)

def zip_file(dirs, target_dir_name):
    """Zip all the contents in the given file"""
    target_path = "{0}/{1}".format(dirs, target_dir_name)
    output_file = "{0}.zip".format(target_path)
    if not os.path.exists(output_file):  # check if the zip file exists
        with ZipFile(output_file, 'w') as zipObj:
            for root, _, files in os.walk(target_path):
                for file in files:  # traverse all files under the dirs
                    zipObj.write(os.path.join(root, file))
            print('File is zipped as "{0}".'.format(output_file))
    else:
        print("Model parameter file already exists.")

def unzip_file(file_path, output_path):
    """Extract all the contents of the zip file"""
    #ref: https://appdividend.com/2022/01/19/python-unzip/
    if os.path.exists(file_path):  # check if the zip file exists
        with ZipFile(file_path, 'r') as zipObj:
            # extracted files will overwrite the existing files with the same name.
            zipObj.extractall(path=output_path)
            print('File is unzipped to "{0}".'.format(output_path))
    else:
        sys.exit("Model parameter file is not found.")

def read_yaml_file(path: str, file_name: str) -> dict:
    """
    @title: reads a .yaml file and returns its content as a dictionary
    @input: path (str): directory path; file_name (str): filename (without file extension).
    @returns: dict: contents of .yaml file
    @reference: https://github.com/stadlmax/Graph-Posterior-Network/tree/main/gpn/utils
    @example: config = read_yaml_file('./configs/xx', 'yaml_file_name')
    """
    file_name = file_name.lower()
    file_path = os.path.join(path, f'{file_name}.yaml')
    if not os.path.exists(file_path):  # check the file path
        raise AssertionError(f'"{file_name}"" file is not found in {path}!')
    with open(file_path) as file:  # open the file path
        yaml_file = yaml.safe_load(file)
    if yaml_file is None:
        yaml_file = {}
    return yaml_file

def split_masks(Y, cut_rate=[0.1, 0.2], seed=42):
    """
    @topic: Split the train/val/test masks
    @input: Y: real label; cut_rate: the cur ratio of train/validation mask.
    @return: label masks (train_mask, val_mask, test_mask).
    """

    def create_mask(shape):
        # Create a zero tensor for mask
        return torch.zeros([shape], dtype=torch.bool)

    # Create masks
    tensor_shape = Y.shape[0]
    train_mask, val_mask, test_mask = create_mask(tensor_shape), create_mask(tensor_shape), create_mask(tensor_shape)
    # Generate a random idx
    torch.manual_seed(seed)
    idx = list(torch.utils.data.RandomSampler(range(0, Y.shape[0])))
    # Split the mask
    train_cut_pos, valid_cut_pos = cut_rate[0], cut_rate[0]+cut_rate[1]
    train_mask[idx[ : int(len(idx)*train_cut_pos)]] = True
    val_mask[idx[int(len(idx)*train_cut_pos) : int(len(idx)*valid_cut_pos)]] = True
    test_mask[idx[int(len(idx)*valid_cut_pos) : int(len(idx)*1)]] = True
    return train_mask, val_mask, test_mask

def generate_random_noise_label(label, noisy_ratio=0.3, seed=42):
    """
    @topic: Randomly generate noise label with given noisy_ratio.
    @input: lable(1D-array), noise_ratio(float), seed(int).
    @return: noisy label (1D-array).
    """
    # generate random pseudo labels
    np.random.seed(seed)
    label_ = np.random.randint(min(label), high=max(label), size=len(label))
    # non-repeatedly select "int(noise_ratio*len(label))" integers as the index of mask in the range of [0, max(label)].
    mask_idx = np.random.choice(len(label), int(noisy_ratio*len(label)), replace=False)
    # replace the values with noise
    label = np.array(label)
    label[mask_idx] = label_[mask_idx]
    return label

def compute_accuracy(logits, labels, mask):
    """Compute the accuracy"""
    logits = logits[mask]
    labels = labels[mask]
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def compute_f1_score(logits, labels, mask, avg='macro'):
    """Compute the f1 score"""
    logits = logits[mask]
    _, y_pred = torch.max(logits, dim=1)
    y_true = labels[mask]
    return f1_score(y_true, y_pred, average=avg)

def compute_entropy(logits, mask):
    """Compute the normalized entropy"""
    logits = logits[mask]
    cat_dist = torch.nn.functional.softmax(logits, dim=1)  # could also use .topk(1, dim = 1) to find the max
    total_entropy = entropy(cat_dist.detach().numpy(), axis=1)  # [0, log(N)]
    normalized_entropy = total_entropy / np.log(cat_dist.shape[1])  # a.k.a. efficiency
    return np.mean(normalized_entropy), np.std(normalized_entropy)

def dist_pre_class(logits, labels, mask):
    """Return the distribution of each class"""
    logits = logits[mask]
    _, y_pred = torch.max(logits, dim=1)
    frq_pred = torch.bincount(y_pred)
    y_true = labels[mask]
    frq_true = torch.bincount(y_true)
    return frq_true.numpy(), frq_pred.numpy()

def save_checkpoint(state, is_best, directory, filename='checkpoint.pth.tar'):
    """Save checkpoint"""
    import shutil
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory+'model_best.pth.tar')

def load_checkpoint(checkpoint_fpath, model, optimizer):
    """Load checkpoint"""
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc = checkpoint['best_acc']
    #best_loss = checkpoint['best_loss']
    return model, optimizer, checkpoint['epoch'], best_acc
    #return model, optimizer, checkpoint['epoch'], best_loss

def count_parameters(model):
    """Count the number of trianable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
