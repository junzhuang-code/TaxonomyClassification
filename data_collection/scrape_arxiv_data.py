#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@topic: Search arXiv papers based on the custom query or paper ids.
@author: Jun Zhuang, Casey Kennington
@ref:
    https://lukasschwab.me/arxiv.py/arxiv.html
    https://github.com/lukasschwab/arxiv.py
    https://github.com/Mahdisadjadi/arxivscraper
"""

import argparse
import arxiv
import pandas as pd
from utils import check_mkdirs


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--search_type", type=str, default='query', help='The type of search: query or id_list.')
parser.add_argument("--max_results", type=int, default=100, help='The max results.')
parser.add_argument("--db_file_name", type=str, default='survey_data.csv', help='Survey data csv file.')
parser.add_argument("--out_file_name", type=str, default='survey_raw.csv', help='Unprocessed surveys csv file.')
args = parser.parse_known_args()[0]

# Setup the search params
# https://info.arxiv.org/help/api/user-manual.html#query_details
RegEx = r'abs/([^v]+)'  # find out the paper id
keywords = ['ti:Large Language Model', 'ti:Survey']
#keywords = ['ti:Large Language Model', 'ti:Review']
search_query = ' AND '.join(keywords)  # 'ti:Large Language Model AND ti:Survey'
search_params = {'query': search_query,
                 'id_list': [],
                 'max_results': args.max_results}

# Define the attributes for collection {header: function_name}
# https://lukasschwab.me/arxiv.py/arxiv.html#Result
attr_dict = {"Title": "title",
             "Authors": "authors",
             "Release Date": "published",
             "Links": "entry_id",
             "Categories": "categories",
             "Summary": "summary",
             #"Comment": "comment",
             }
search_params['attr'] = attr_dict

# Define the path
data_path = "../data/survey_data/"
check_mkdirs(data_path)
db_file_path = data_path+args.db_file_name
out_file_path = data_path+args.out_file_name


def search_arxiv_papers(search_type: str, search_params: dict) -> pd.DataFrame:
    """Search arXiv papers."""
    # Construct the default API client.
    client = arxiv.Client()
    assert search_type in ['query', 'id_list']
    if search_type == 'query':
        search = arxiv.Search(
          query = search_params['query'],
          max_results = search_params['max_results'],
          sort_by = arxiv.SortCriterion.Relevance,
          sort_order = arxiv.SortOrder.Descending
        )
    if search_type == 'id_list':
        search = arxiv.Search(
          id_list = search_params['id_list'],
          max_results = search_params['max_results'],
          sort_by = arxiv.SortCriterion.Relevance,
          sort_order = arxiv.SortOrder.Descending
        )
    results = client.results(search)
    headers = list(search_params['attr'].keys())
    paper_list = []
    for result in results:  # traverse each paper
        paper_attr = ['' for _ in range(len(headers))]
        for i, attr_name in enumerate(search_params['attr'].values()):  # traverse each attribute
            attr_value = getattr(result, attr_name, None)
            paper_attr[i] = attr_value
        paper_list.append(paper_attr)
    return pd.DataFrame(paper_list, index=None, columns=headers)


def get_paper_ids(db_file_path: str, identifier_col: str ='Paper ID') -> list:
    """Get paper ids (list) from a given database."""
    id_df = pd.read_csv(db_file_path, dtype={identifier_col: str})
    id_list = list(id_df[identifier_col].dropna().tolist())
    return id_list


def process_collected_results(results_df: pd.DataFrame, id_set: set) -> pd.DataFrame:
    """Process the collected results."""
    # Extract the paper id and move to the 1st column
    results_df['Paper ID'] = results_df['Links'].str.extract(RegEx).astype(str)
    results_df = results_df[['Paper ID'] + [col for col in results_df.columns if col != 'Paper ID']]
    # Filter out rows where "Paper ID" is in the specified set
    results_df = results_df[~results_df['Paper ID'].isin(id_set)]
    # Extract author's name as a string
    results_df['Authors'] = results_df['Authors'].apply(lambda authors: 
                            ', '.join([author.name for author in authors]))
    # Modify the datetime format
    results_df['Release Date'] = results_df['Release Date'].dt.strftime('%Y-%m-%d')
    # Join the categories as a string
    results_df['Categories'] = results_df['Categories'].apply(lambda categories: 
                                ', '.join(categories))
    # Remove the \n symbol
    results_df['Summary'] = results_df['Summary'].apply(lambda summary: 
                                    summary.replace('\n', ' '))
    return results_df


if __name__ == '__main__':
    # Get the paper id list
    id_list = get_paper_ids(db_file_path, 'Paper ID')
    # Set up the params if necessary
    if args.search_type == 'query':
        duplicate_check = set(id_list)
    if args.search_type == 'id_list':
        search_params['id_list'] = id_list
        duplicate_check = set()
    # Search papers based on the given params
    res_df = search_arxiv_papers(args.search_type, search_params)
    # Process the collected results.
    paper_df = process_collected_results(res_df, duplicate_check)
    # Save the results
    paper_df.to_csv(out_file_path, index=False)
