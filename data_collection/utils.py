#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@topic: Utils.
@author: Jun Zhuang, Casey Kennington
"""

import os
import csv

def read_csv_file(file_path, encoding_type='utf-8'):
    with open(file_path, 'r', newline='', encoding=encoding_type) as csv_file:
        csv_reader = csv.reader(csv_file)
        data = list(csv_reader)
    return data

def write_csv_file(data, file_path, encoding_type='utf-8'):
    with open(file_path, 'w', newline='', encoding=encoding_type) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)

def read_txt_file(file_path, encoding_type='utf-8'):
    with open(file_path, 'r', encoding=encoding_type) as file:
        text_data = file.read()
    return text_data

def write_file(data, file_path, encoding_type='utf-8'):
    with open(file_path, 'w', encoding=encoding_type) as file:
        file.write(data)

def check_mkdirs(dir_name):
    """Create a data path if necessary."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
