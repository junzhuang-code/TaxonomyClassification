#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Understanding Survey Paper Taxonomy about Large Language Models via Graph Representation Learning
@topic: Train the GNN model and evaluation.
@author: Jun Zhuang, Casey Kennington
"""

import os
import time
import argparse
from datetime import datetime
import numpy as np
import torch
from models_GNN import GNNs
from load_data import LoadDataset
from sklearn.model_selection import train_test_split
from utils import read_yaml_file, read_pickle, dump_pickle, zip_file, check_mkdirs, \
                    build_bool_mask, split_masks, generate_random_noise_label, \
                    load_checkpoint, save_checkpoint, count_parameters, \
                    compute_accuracy, compute_f1_score, dist_pre_class, compute_entropy
from torch.utils.tensorboard import SummaryWriter

# Define the arguments
parser = argparse.ArgumentParser(description="Read arguments for training.")
parser.add_argument("--data_id", type=str, default='1', help="dataset ID.")
parser.add_argument("--data_name", type=str, default="sv_cograph0", help="The name of dataset.",
                    choices=["sv_textgraph", "sv_cograph0"])
parser.add_argument("--model_name", type=str, default="GCN", help="Graph operator.")
parser.add_argument("--NOISE_RATIO", type=float, default=0.0, help="Noise ratio.")
parser.add_argument("--NUM_EPOCHS", type=int, default=500, help="The number of training epochs.")
parser.add_argument("--is_trainable", type=bool, default=True, help="Train the model or not.")
parser.add_argument("--GPU", type=int, default=-1, help="Input GPU device id or -1.")
parser.add_argument("--config_file", type=str, default="graph_models", help="The name of config files.")
args = parser.parse_known_args()[0]

# Read config files
if len(args.config_file) > 0:
    config_file = read_yaml_file("../../config", args.config_file)
    train_config = config_file["train_model"]
    args.data_id = train_config["data_id"]
    args.data_name = train_config["data_name"]
    args.model_name = train_config["model_name"]
    args.NOISE_RATIO = train_config["NOISE_RATIO"]
    args.NUM_EPOCHS = train_config["NUM_EPOCHS"]
    args.is_trainable = train_config["is_trainable"]
    args.GPU = train_config["GPU"]

# Path for input text/co-graph
sv_cograph_name_dict = {"sv_cograph0": f"G_coauthor{args.data_id}.pkl",
                        "sv_cograph1": f"G_cocategory{args.data_id}_all.pkl",
                        "sv_cograph2": f"G_cocategory{args.data_id}_removed_cs.CL.pkl",
                        "sv_cograph3": f"G_cocategory{args.data_id}_removed_cs.AI.pkl",                        
                        "sv_cograph4": f"G_cocategory{args.data_id}_removed_cs.CL_cs.AI.pkl",
                        "sv_cograph5": f"G_cocategory{args.data_id}_removed_cs.IR.pkl",
                        "sv_cograph6": f"G_cocategory{args.data_id}_removed_cs.RO.pkl",
                        "sv_cograph7": f"G_cocategory{args.data_id}_removed_cs.SE.pkl",
                        "sv_cograph8": f"G_cocategory{args.data_id}_removed_cs.IR_cs.RO.pkl",
                        "sv_cograph9": f"G_cocategory{args.data_id}_removed_cs.IR_cs.SE.pkl",
                        "sv_cograph10": f"G_cocategory{args.data_id}_removed_cs.RO_cs.SE.pkl",
                        "sv_cograph11": f"G_cocategory{args.data_id}_removed_cs.IR_cs.RO_cs.SE.pkl",
                        }
if args.data_name == 'sv_textgraph':
    args.data_name = f'llm_survey{args.data_id}'
    data_path = '../../data/text_corpus/'
    graph_path = data_path+'text_graph'
    label_path = data_path+'labels'
if args.data_name in list(sv_cograph_name_dict.keys()):
    data_path = '../../data/graph_data/'
    label_path = data_path+'labels'
check_mkdirs(label_path)
# Path for saving the parameters
dirs = 'runs/{0}_{1}/'.format(args.data_name, args.model_name)
path = dirs + 'model_best.pth.tar'
dirs_attack = '../../data/text_corpus/parameters/'
# Initialize the parameters
seed = 0
is_text_graph = False
if args.data_name in [f"llm_survey{args.data_id}"]:
    is_text_graph = True
    VAL_RATE = 0.2  # the percentage of validation set split from the training set. 0.2
    LR = 0.02  # for text graph
else:
    CUT_RATE = [0.1, 0.2]  # the split ratio of train/validation mask.
    LR = 0.001
    if args.data_name in list(sv_cograph_name_dict.keys()):
        CUT_RATE = [0.6, 0.2]
        LR = 0.01
N_HIDDEN = 200
DROPOUT = 0.5
WEIGHT_DECAY = 0
kwargs_dicts = {"GCN": [None, None],  # aggregator_type, n_filter
                "GraphSAGE": ["gcn", None],
                "GIN": ["mean", None],
                "TAGCN": [None, 3],
                }


def train(model, opt, dirs, train_mask, val_mask):
    """
    @topic: Fitting the model.
    @input: model, train/val masks.
    @return: train and save the model parameters.
    """
    # Define the loss function
    loss_ce = torch.nn.CrossEntropyLoss()
    # Load checkpoint
    try:
        model, opt, start_epoch, best_acc = \
            load_checkpoint(dirs+'model_best.pth.tar', model, opt)
    except:
        print("Model parameter is not found.")
        start_epoch = 1
        best_acc = 0
    # Obtain the feat & label
    feat, label = model.graph.ndata['feat'], model.graph.ndata['label']
    feat = feat.to_dense() if feat.is_sparse and args.GPU < 0 else feat
    # Train with additional epochs if given #epochs < trained #epochs.
    if args.NUM_EPOCHS <= start_epoch:
        args.NUM_EPOCHS += start_epoch
    # Used for tensorboard
    writer = SummaryWriter(log_dir=dirs, 
            comment="_time%s"%(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), 
            purge_step=start_epoch)
    dur = []
    for epoch in range(start_epoch, args.NUM_EPOCHS):
        model.train()  # change the mode of dropout & batch normalization to avoid overfitting.
        if epoch >= 3:  # record the runtime
            t0 = time.time()
        # Forward
        logits, _ = model.forward(feat)
        # Compute the cross entropy between predicted labels and supervised labels.
        loss = loss_ce(logits[train_mask], label[train_mask])
        # Backward
        opt.zero_grad()  # reset the gradient to avoid accumulation.
        loss.backward()  # compute the gradient
        opt.step()  # Update the weights (W_{new} = W_{old} - ∂(Gradient))
        if epoch >= 3:
            dur.append(time.time() - t0)
        # Compute the acc/loss for evaluation
        with torch.no_grad():
            acc_train = compute_accuracy(logits, label, train_mask)
            acc_val = compute_accuracy(logits, label, val_mask)
            loss_train = loss_ce(logits[train_mask], label[train_mask])
            loss_val = loss_ce(logits[val_mask], label[val_mask])
        # Define the file name
        FileName = "Epoch{0}.pth.tar".format(epoch)
        # Delete previous existing parameter file (optional)
        if os.path.exists(dirs+"Epoch{0}.pth.tar".format(epoch-1)):
            os.remove(dirs+"Epoch{0}.pth.tar".format(epoch-1))
        # Update the parameters
        #if acc_val > best_acc:
        if acc_val > best_acc or epoch % 10 == 0:
            best_acc = acc_val
            is_best = True
            # Save checkpoint
            save_checkpoint(state = {
                                    'epoch': epoch + 1,
                                    'state_dict': model.state_dict(),
                                    'best_acc': best_acc,
                                    'optimizer': opt.state_dict()
                                    }, \
                            is_best = is_best, \
                            directory = dirs, \
                            filename = FileName
                            )
        # Output the results & update summaryWriter
        print("Epoch {:05d} | Time(s) {:.4f} | Train Loss: {:.4f} | Val Loss: {:.4f} "\
            "| Train Accuracy: {:.4f} | Val Accuracy: {:.4f}"\
            .format(epoch, np.mean(dur), \
                    loss_train.item(), loss_val.item(), acc_train, acc_val))
        writer.add_scalar('Loss/train', loss_train.item(), epoch)
        writer.add_scalar('Loss/val', loss_val.item(), epoch)
        writer.add_scalar('Accuracy/train', acc_train, epoch)
        writer.add_scalar('Accuracy/val', acc_val, epoch)
        writer.flush()
    writer.close()


def prediction(model, opt, path, graph):
    """
    @topic: Generate predicted labels with well-trained GCN model.
    @input: model, graph.
    @return: predicted labels (1D Tensor).
    """
    model.eval()
    #model, opt, _, _ = load_checkpoint(path, model, opt)
    model.graph = graph
    feat = model.graph.ndata['feat']
    feat = feat.to_dense() if feat.is_sparse and args.GPU < 0 else feat
    if not graph.number_of_nodes() == len(feat):
        return "The length of adj and feat is not equal!"
    logits, node_embs_pred = model(feat)
    Y_pred_2d_softmax = torch.nn.functional.softmax(logits, dim=1)  # Normalize each row to sum=1
    Y_pred = torch.max(Y_pred_2d_softmax, dim=1)[1]  # predicted label (1d)
    return Y_pred, node_embs_pred


def evaluation(model, opt, path, graph, test_mask):
    """
    @topic: Evaluation on the given model.
    @input: model, graph, and test mask.
    @return: print out the test acc/f1/ent.
    """
    model.eval()  # fix the DO & BN during evaluation
    #model, opt, _, _ = load_checkpoint(path, model, opt)
    model.graph = graph
    feat, label = model.graph.ndata['feat'], model.graph.ndata['label']
    feat = feat.to_dense() if feat.is_sparse and args.GPU < 0 else feat
    if not graph.number_of_nodes() == len(feat) == len(label) == len(test_mask):
        return "The length of adj/feat/label/test_mask is not equal!"
    logits, _ = model(feat)
    if args.GPU >= 0:
        logits = logits.cpu()
        label = label.cpu()
        test_mask = test_mask.cpu()
    acc = compute_accuracy(logits, label, test_mask)  # the higher the better
    print("Best Testing Accuracy: {:.2%}.".format(acc))
    f1 = compute_f1_score(logits, label, test_mask, avg='weighted')
    print("Test F1 Score: {:.2%}.".format(f1))
    dist_true, dist_pred = dist_pre_class(logits, label, test_mask)
    print("The distributions of groundtruth classes: \n {0}".format(dist_true))
    print("The distributions of predicted classes: \n {0}".format(dist_pred))
    ent_mean, ent_std = compute_entropy(logits, test_mask)  # the lower the better
    print("The normalized entropy: {:.2%}(±{:.2%}).".format(ent_mean, ent_std))


if __name__ == "__main__":
    print("Load dataset and preprocessing.")
    if args.data_name in list(sv_cograph_name_dict.keys()):  # Input llm survey co-graphs
        graph = read_pickle(data_path+sv_cograph_name_dict[args.data_name])
    else:  # Input text graphs
        data = LoadDataset(args.data_name)
        graph = data.load_text_graph_data(graph_path, label_path)
        train_list, test_list = data.train_lst, data.test_lst

    # Build train/val/test masks
    label = graph.ndata['label']
    print("Class ID: ", set(label.numpy()))
    if is_text_graph:  # for text graph
        train_id, val_id = train_test_split(train_list, test_size=VAL_RATE, 
                                            shuffle=True, random_state=seed)
        train_mask = build_bool_mask(label, train_id)
        val_mask = build_bool_mask(label, val_id)
        test_mask = build_bool_mask(label, test_list)
    else:  # for graph
        train_mask, val_mask, test_mask = split_masks(label, CUT_RATE, seed=seed)
    # Build the noise labels (maintain closed-set noise in the train labels)    
    Y_noisy_tv = generate_random_noise_label(label[train_mask+val_mask], 
                                             noisy_ratio=args.NOISE_RATIO, seed=seed)
    Y_noisy_tv = torch.LongTensor(Y_noisy_tv)
    Y_noisy = label.clone()
    Y_noisy[train_mask+val_mask] = Y_noisy_tv
    graph.ndata['label'] = Y_noisy
    # Display the variables
    print("-----Data statistics-----\n"
          f"Number of Nodes: {graph.number_of_nodes()}\n"
          f"Number of Edges: {graph.number_of_edges()}\n"
          f"Number of Features: {graph.ndata['feat'].shape[1]}\n"
          f"Number of Classes: {len(torch.unique(label))}"
          )

    # Initialize the model
    model = GNNs(graph,
                in_feats=graph.ndata['feat'].shape[1],
                n_hidden=N_HIDDEN,
                n_classes=len(torch.unique(label)),
                activation=torch.nn.functional.relu,
                dropout=DROPOUT,
                model_name=args.model_name,
                aggregator_type=kwargs_dicts[args.model_name][0],
                n_filter=kwargs_dicts[args.model_name][1],
                )
    print(f'The model has {count_parameters(model):,} trainable parameters.')
    if args.GPU >= 0:  # if gpu is available
        print("Using GPU!")
        torch.cuda.set_device(args.GPU)
        model.graph = model.graph.to('cuda')
        model.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
    else:
        print("Using CPU!")
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Training the model
    print("Training on train graphs ...")
    if args.is_trainable:
        train(model, opt, dirs, train_mask, val_mask)
        zip_file('./runs', '{0}_{1}'.format(args.data_name, args.model_name))

    # Evaluation
    print("Evaluation on test graphs:")
    # We don't apply evasion attacks here so we use the same graph.
    evaluation(model, opt, path, model.graph, test_mask)
    # Generate and save predicted labels
    if args.data_name in list(sv_cograph_name_dict.keys()):
        graph = graph.to('cuda') if args.GPU >= 0 else graph
        Y_pred, node_embs = prediction(model, opt, path, graph)  # prediction on all nodes
        if args.GPU >= 0:
            Y_pred, node_embs = Y_pred.cpu(), node_embs.cpu()
        dump_pickle(f"{label_path}/Y_preds{sv_cograph_name_dict[args.data_name][1:]}", [Y_pred, node_embs])
        print(f"Y_pred's shape: {Y_pred.shape}, node_embs' shape: {node_embs.shape}.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
