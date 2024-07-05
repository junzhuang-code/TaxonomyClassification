#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Understanding Survey Paper Taxonomy about Large Language Models via Graph Representation Learning
@topic: Utils modules.
@author: Jun Zhuang, Casey Kennington
"""

import os
import pickle
import yaml
import torch
from torch.utils.data import Dataset


class MyData(Dataset):
    def __init__(self, text_tokens, labels):
        self.text_tokens = text_tokens
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.text_tokens.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


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

def save_checkpoint(state, is_best, directory, filename='checkpoint.pth.tar'):
    """Save checkpoint."""
    import shutil
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory+'model_best.pth.tar')

def load_checkpoint(checkpoint_fpath, model, optimizer):
    """Load checkpoint."""
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc = checkpoint['best_acc']
    return model, optimizer, checkpoint['epoch'], best_acc

def read_pickle(file_name):
    """Load the dataset"""
    with open (file_name,'rb') as file:
        return pickle.load(file)
