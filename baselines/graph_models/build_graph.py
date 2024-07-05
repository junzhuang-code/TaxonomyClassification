#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Understanding Survey Paper Taxonomy about Large Language Models via Graph Representation Learning
@topic: Build the coauthor/cocategory graphs.
@author: Jun Zhuang, Casey Kennington
"""

import argparse
import dgl
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import dump_pickle, check_mkdirs

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_file_name", type=str, default='survey_data1.csv', help='data file name.')
parser.add_argument("--graph_type", type=str, default="cocategory", help='The type of graphs.',
                    choices=['coauthor', 'cocategory'])
parser.add_argument("--is_plot", type=bool, default=True, help='Plot the graph or Save data.')
args = parser.parse_known_args()[0]

# Define the path
data_id = args.data_file_name[len("survey_data"):-len(".csv")]
data_dirs = '../../data/survey_data/'
out_dirs = '../../data/graph_data/'
fig_dirs = '../../data_analysis/figures/'
for dirs in [out_dirs, fig_dirs]:
    check_mkdirs(dirs)
# Setup the skip sets
skip_set_dict = {"Empty": {},
                 "CL": {'cs.CL'},
                 "AI": {'cs.AI'},
                 "CL_AI": {'cs.CL', 'cs.AI'},
                 "SE": {'cs.SE'},
                 "RO": {'cs.RO'},
                 "IR": {'cs.IR'},
                 "SE_RO": {'cs.SE', 'cs.RO'},
                 "SE_IR": {'cs.SE', 'cs.IR'},
                 "RO_IR": {'cs.RO', 'cs.IR'},
                 "SE_RO_IR": {'cs.SE', 'cs.RO', 'cs.IR'},
                 }


def build_coauthor_graph(authors_df, taxonomy_df, is_selfloop=True):
    """Build a coauthor graph based on author list and taxonomy."""
    # Create a nx graph
    G = nx.Graph()
    # Add nodes (paper) to the graph
    for index, row in enumerate(taxonomy_df):
        G.add_node(index, taxonomy=row)
    # Split the 'Authors' column to create a list of authors for each paper
    authors_df = authors_df.apply(lambda x: x.split(', '))
    # Add edges based on author collaborations
    for i, authors_i in enumerate(authors_df):
        for j, authors_j in enumerate(authors_df[i+1:]):
            # Add a self-connected edge
            if i == j and is_selfloop:
                G.add_edge(i, j)
            # Add a edge if we found common authors
            common_authors = set(authors_i).intersection(set(authors_j))
            if common_authors and i != j:
                G.add_edge(i, j)
    return G


def build_cocategory_graph(category_df, taxonomy_df, skip_set, is_selfloop=True):
    """Build a cocategory graph based on category lists and taxonomy."""
    # Create a nx graph
    G = nx.Graph()
    # Add nodes (category) to the graph
    for index, row in enumerate(taxonomy_df):
        G.add_node(index, taxonomy=row)
    # Split the 'Category' column to create a list of arxiv category for each paper
    # We use pd.notnull(x) to filter out "None" values (non-arxiv papers)
    category_df = category_df.apply(lambda x: x.split(', ') if pd.notnull(x) else [])
    # Add edges based on the category
    for i, category_i in enumerate(category_df):
        for j, category_j in enumerate(category_df[i+1:]):
            # Add a self-connected edge
            if i == j and is_selfloop:
                G.add_edge(i, j)
            # Add a edge if we found common categories
            common_categories = set(category_i).intersection(set(category_j))
            if skip_set: # Filter out some categories given in the skip_list
                common_categories -= skip_set
            if common_categories and i != j:
                G.add_edge(i, j, shared_categories=list(common_categories))
    return G


def build_feature_matrix(data_df):
    """Build a feature matrix."""
    # Vectorize the 'Title' and 'Summary' using TF-IDF
    # TF-IDF特征矩阵是一个稀疏矩阵{行: 文档; 列: 词语}。矩阵中的元素是对应词语在文档中的TF-IDF值。该矩阵将文本数据表示为一个数值矩阵，保留了词语的重要性信息。
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    title_tfidf = tfidf_vectorizer.fit_transform(data_df['Title'])
    summary_tfidf = tfidf_vectorizer.fit_transform(data_df['Summary'])

    # Apply One-hot embedding to 'Categories'
    categories_split = data_df['Categories'].str.split(', ', expand=True)
    categories_encoded = pd.get_dummies(categories_split, prefix='')

    # Combine all the features
    feature_matrix = pd.DataFrame.sparse.from_spmatrix(title_tfidf)
    feature_matrix = pd.concat([feature_matrix, pd.DataFrame.sparse.from_spmatrix(summary_tfidf)], axis=1)
    feature_matrix = pd.concat([feature_matrix, categories_encoded], axis=1)
    return feature_matrix


def plot_coauthor_graph(G, taxonomy_df, title, fig_path, nd=0.5, seed=41):
    """Plot a coauthor graph based on author lists and taxonomy."""
    # Plot the graph with node colors based on taxonomy
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=nd, seed=seed)  # k controls the node distance
    # Get unique taxonomy colors
    taxonomies_set = set(taxonomy for taxonomy in taxonomy_df)
    taxonomy_colors = {taxonomy: plt.colormaps.get_cmap('tab20')(i) 
                       for i, taxonomy in enumerate(taxonomies_set)}
    # Assign colors to nodes based on taxonomy
    node_colors = [taxonomy_colors[G.nodes[node]['taxonomy']] for node in G.nodes]
    # Draw the graph with colored nodes
    nx.draw(G, pos, with_labels=True, font_size=20, node_size=800, 
            node_color=node_colors, font_color='black', alpha=0.8)
    # Draw legend for colors
    legend_labels = {taxonomy: plt.Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor=color, markersize=20, label=taxonomy)
                     for taxonomy, color in taxonomy_colors.items()}
    plt.legend(handles=list(legend_labels.values()), title='Taxonomy', title_fontsize='25', fontsize='20')
    plt.title(title, fontsize='40')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.show()


def plot_cocategory_graph(G, taxonomy_df, title, fig_path, nd=0.5, seed=41):
    """Plot a cocategory graph based on category lists and taxonomy."""
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=nd, seed=seed)  # k controls the node distance
    # Get unique taxonomy colors
    taxonomies_set = set(taxonomy for taxonomy in taxonomy_df)
    taxonomy_colors = {taxonomy: plt.colormaps.get_cmap('tab20')(i) for i, taxonomy in enumerate(taxonomies_set)}
    # Assign colors to nodes based on taxonomy
    node_colors = [taxonomy_colors[G.nodes[node]['taxonomy']] for node in G.nodes]
    # Draw the graph with shared categories as labels
    edge_labels = {(edge[0], edge[1]): ', '.join(G.edges[edge]['shared_categories']) for edge in G.edges}
    nx.draw(G, pos, with_labels=True, font_size=20, node_size=800, 
            node_color=node_colors, font_color='black', alpha=0.8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10)
    # Draw legend for colors
    legend_labels = {taxonomy: plt.Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor=color, markersize=20, label=taxonomy)
                     for taxonomy, color in taxonomy_colors.items()}
    plt.legend(handles=list(legend_labels.values()), title='Taxonomy', title_fontsize='25', fontsize='20')
    plt.title(title, fontsize='40')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.show()


def build_graph(skip_set, is_plot=True):
    """Build a graph."""
    # Read dataset
    input_path = data_dirs + args.data_file_name
    data_df = pd.read_csv(input_path)
    # Build a feature matrix
    feature_matrix = build_feature_matrix(data_df)
    feat_ts = torch.tensor(feature_matrix.values, dtype=torch.float32)
    # Build labels from data_df['Taxonomy']
    label_ts = torch.tensor(data_df['Taxonomy'].astype('category').cat.codes.values, dtype=torch.int64)
    # Build a nx graph
    if args.graph_type == 'coauthor':
        output_path = out_dirs + f"G_{args.graph_type}{data_id}.pkl"
        fig_path = fig_dirs + f"fig_{args.graph_type}{data_id}_graph.pdf"
        G_nx = build_coauthor_graph(data_df['Authors'], data_df['Taxonomy'], is_selfloop=False)
        #title = 'Visualization of Collaborations in A Co-author Graph by Taxonomy'
        title = 'Co-author Graphs'
        if is_plot:
            plot_coauthor_graph(G_nx, data_df['Taxonomy'], title, fig_path, nd=0.5, seed=42)
    if args.graph_type == 'cocategory':
        out_subtitle = f"removed_{'_'.join(skip_set)}" if skip_set else "all"
        output_path = out_dirs + f"G_{args.graph_type}{data_id}_{out_subtitle}.pkl"
        fig_path = fig_dirs + f"fig_{args.graph_type}{data_id}_graph_{out_subtitle}.pdf"  # jpg
        G_nx = build_cocategory_graph(data_df['Categories'], data_df['Taxonomy'],
                                   skip_set, is_selfloop=False)
        # set up the title before plotting
        sub_title = f'(Removed {", ".join(skip_set)})' if skip_set else '(All Categories)'
        #title = "Visualization of Co-category in Graphs by Taxonomy " + sub_title
        title = "Co-category Graphs " + sub_title
        if is_plot:
            plot_cocategory_graph(G_nx, data_df['Taxonomy'], title, fig_path, nd=0.5, seed=42)
    # Save the graph data
    if not is_plot:
        G_dgl = dgl.from_networkx(G_nx)
        G_dgl = dgl.add_self_loop(G_dgl)
        G_dgl.ndata['feat'] = feat_ts
        G_dgl.ndata['label'] = label_ts
        dump_pickle(output_path, G_dgl)


if __name__ == '__main__':
    if args.graph_type == 'coauthor':
        build_graph({}, args.is_plot)
    if args.graph_type == 'cocategory':
        for skip_set in skip_set_dict.values():
            build_graph(skip_set, args.is_plot)
