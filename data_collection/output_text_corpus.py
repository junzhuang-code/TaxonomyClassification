#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@topic: Output the text corpus as txt/csv files.
@author: Jun Zhuang, Casey Kennington
"""

import argparse
import pandas as pd
from utils import check_mkdirs
    

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_file_name", type=str, default='survey_data1.csv', help='Survey data csv file.')
parser.add_argument("--data_format", type=str, default='txt', help='Output data format: txt, csv.')
args = parser.parse_known_args()[0]

data_id = args.input_file_name[len("survey_data"):-len(".csv")]
TEST_RATIO = 0.2
split_col_name = 'Split'
# Define the path
data_path = '../data/survey_data/'
output_data_path = '../data/text_corpus/corpus/'  # for textgnn
output_labels_path = '../data/text_corpus/labels/'
for path in [data_path, output_data_path, output_labels_path]:
    check_mkdirs(path)
input_file_path = data_path+args.input_file_name
dformat2filename_dict = {"txt": f"llm_survey{data_id}.txt",
                         "csv": f"llm_survey{data_id}.csv",
                        }
out_data_path = output_data_path + dformat2filename_dict[args.data_format]
out_labels_path = output_labels_path + dformat2filename_dict[args.data_format]


def label_data(data_df, split_col_name, TEST_RATIO):
    """Label the randomized data frame."""
    # Randomize the instances
    data_df_rdn = data_df.sample(frac=1, random_state=42)
    # Split the train & test data
    test_size = int(TEST_RATIO * len(data_df_rdn))
    #train_set = data_df_rdn.head(len(data_df_rdn) - test_size)
    test_set = data_df_rdn.tail(test_size)
    # Add the labels
    data_df_rdn[split_col_name] = 'train'
    data_df_rdn.loc[test_set.index, split_col_name] = 'test'
    return data_df_rdn

def output_corpus_txt():
    """Output the text corpus to txt files."""
    # Read data
    data_df = pd.read_csv(input_file_path)
    # Process data
    data_df = label_data(data_df, split_col_name, TEST_RATIO)  # label the data
    data_df['Text'] = data_df['Title']+ ' ' + data_df['Summary']  # merge text data
    data_df['Text'] = data_df['Text'].apply(
        lambda x: ''.join(char for char in x if ord(char) <= 127)) # remove non-ASCII char
    #labels_df = data_df[[split_col_name, 'Taxonomy']].reset_index(drop=True)  # build labels
    labels_df = data_df[[split_col_name, 'Taxonomy']]  # build labels & keep the original index
    # Output the text corpus and labels
    data_df['Text'].to_csv(out_data_path, index=False, header=False)
    labels_df.to_csv(out_labels_path, sep='\t', index_label='Index', header=False)

def output_corpus_csv():
    """Output the text corpus to csv file."""
    # Read data
    data_df = pd.read_csv(input_file_path)
    # Process data
    data_df['Text'] = data_df['Title']+ ' ' + data_df['Summary']  # merge text data
    data_df['Text'] = data_df['Text'].apply(
        lambda x: ''.join(char for char in x if ord(char) <= 127)) # remove non-ASCII char
    text_df = data_df[['Taxonomy', 'Text']]  # slice the data and labels
    text_df.to_csv(out_data_path, index=False, header=True)

def output_csv():
    """Output the data to a csv file for testing purposes."""
    # Read data
    data_df = pd.read_csv(input_file_path)
    # Process data
    data_df = data_df.sample(frac=1, random_state=42)  # Shuffle the data
    text_df = data_df[['Taxonomy', 'Title', 'Summary']]
    text_df.to_csv(out_data_path, index=False, header=True)


if __name__ == '__main__':
    if args.data_format == 'txt':
        output_corpus_txt()
    if args.data_format == 'csv':
        #output_corpus_csv()
        output_csv()
    