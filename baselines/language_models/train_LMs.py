#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Understanding Survey Paper Taxonomy about Large Language Models via Graph Representation Learning
@topic: Implement language models for text classification.
@author: Jun Zhuang, Casey Kennington
@dependences:
    conda install transformers, sentencepiece
@ref:
    https://huggingface.co/docs/transformers/tasks/sequence_classification
    https://www.shecodes.io/athena/92466-how-to-fine-tune-llama-for-text-classification
"""

import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import LlamaTokenizerFast, LlamaForSequenceClassification
from utils import MyData, read_yaml_file, load_checkpoint, save_checkpoint
warnings.filterwarnings("ignore")

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_file_name", type=str, default='llm_survey1.txt', help='text data file.')
parser.add_argument("--model_name", type=str, default='DistilBERT', help='Model name.',
                    choices=['BERT', 'RoBERTa', 'DistilBERT', 'XLNet', 'Electra', 'Albert', 'BART', 'DeBERTa', 'Llama2'])
parser.add_argument("--access_token", type=str, default="", help='HuggingFace token.')
parser.add_argument("--NUM_EPOCHS", type=int, default=30, help='The number of training epochs.')
parser.add_argument("--BATCH_SIZE", type=int, default=16, help='The number of batch size.')
parser.add_argument('--device', type=str, default='cpu', help='device name.',
                    choices=['cpu', 'cuda', 'mps'])
parser.add_argument('--GPU', type=int, default=0, help='gpu id.')
parser.add_argument("--config_file", type=str, default="language_models", help="The name of config files.")
args = parser.parse_known_args()[0]

# Read config files
if len(args.config_file) > 0:
    config_file = read_yaml_file("../../config", args.config_file)
    train_config = config_file["train_model"]
    args.data_file_name = train_config["data_file_name"]
    args.model_name = train_config["model_name"]
    args.access_token = train_config["access_token"]
    args.NUM_EPOCHS = train_config["NUM_EPOCHS"]
    args.BATCH_SIZE = train_config["BATCH_SIZE"]
    args.device = train_config["device"]
    args.GPU = train_config["GPU"]

seed = 0
VAL_RATIO, TEST_RATIO = 0.2, 0.2
LR = 1e-4
is_ignore = True if args.model_name in ['Albert'] else False
model_type_dict = {"BERT": "bert-base-uncased",
                   "RoBERTa": "roberta-base",
                   "DistilBERT": "distilbert-base-uncased",
                   "XLNet": "xlnet-base-cased",
                   "Electra": "google/electra-base-discriminator",
                   "Albert": "textattack/albert-base-v2-imdb",
                   "BART": "facebook/bart-base",
                   "DeBERTa": "microsoft/deberta-base",
                   "Llama2": "meta-llama/Llama-2-7b-hf",
                   }
# Define the path
data_dirs = '../../data/text_corpus/'
data_path = data_dirs+'clean_corpus/'
labels_path = data_dirs+'labels/'
ckpt_dirs = f"./runs/{args.data_file_name[:-len('.txt')]}_{args.model_name}/"
ckpt_path = ckpt_dirs + 'model_best.pth.tar'


def train(model, optimizer, ckpt_dirs, train_loader, val_loader, num_epochs, device):
    """Train the model."""
    # Load checkpoint
    try:
        model, optimizer, start_epoch, best_acc = \
            load_checkpoint(ckpt_dirs+'model_best.pth.tar', model, optimizer)
    except:
        print("Model parameter is not found.")
        start_epoch = 1
        best_acc = 0
    # Train with additional epochs if given #epochs < trained #epochs.
    if num_epochs <= start_epoch:
        num_epochs = start_epoch + 1
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss_train, total_loss_val = 0, 0
        start_time = time.time()
        for tbatch in train_loader:
            if torch.cuda.is_available() and device == 'cuda':
                tbatch = {key: val.to(device) for key, val in tbatch.items()}
            outputs_train = model(tbatch['input_ids'], attention_mask=tbatch['attention_mask'], 
                            labels=tbatch['labels'])
            loss_train = outputs_train.loss
            total_loss_train += loss_train.item()
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
        # Compute the loss for validation
        model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for vbatch in val_loader:
                if torch.cuda.is_available() and device == 'cuda':
                    vbatch = {key: val.to(device) for key, val in vbatch.items()}
                outputs_val = model(vbatch['input_ids'], attention_mask=vbatch['attention_mask'], 
                                labels=vbatch['labels'])
                loss_val = outputs_val.loss
                total_loss_val += loss_val.item()
                y_pred.extend(torch.argmax(outputs_val.logits, dim=1).cpu().numpy())
                y_true.extend(vbatch['labels'].cpu().numpy())
        # Compute the valid accuracy
        acc_val = accuracy_score(y_true, y_pred)        
        # Delete previous existing parameter file (optional)
        if os.path.exists(ckpt_dirs+f"Epoch{epoch-1}.pth.tar"):
            os.remove(ckpt_dirs+f"Epoch{epoch-1}.pth.tar")
        # Update the parameters and save the weights
        if acc_val > best_acc:
            best_acc = acc_val
            is_best = True
            # Save checkpoint
            save_checkpoint(state = {'epoch': epoch + 1,
                                     'state_dict': model.state_dict(),
                                     'best_acc': best_acc,
                                     'optimizer': optimizer.state_dict()
                                    }, \
                            is_best = is_best, \
                            directory = ckpt_dirs, \
                            filename = f"Epoch{epoch}.pth.tar"
                            )
            print("Save the best model in the {0} epoch.".format(epoch))
        epoch_time = time.time() - start_time
        avg_loss_train = total_loss_train / len(train_loader)
        avg_loss_val = total_loss_val / len(val_loader)
        print(f"Epoch {epoch+1}/{args.NUM_EPOCHS} - Valid Acc: {acc_val:.2%} "\
              f"Train Loss: {avg_loss_train:.4f} - Valid Loss: {avg_loss_val:.4f} - "\
              f"Time: {epoch_time:.2f}s")

def evaluation(model, opt, ckpt_path, test_loader, device):
    """Evaluation."""
    model.eval()
    model, _, _, _ = load_checkpoint(ckpt_path, model, opt)
    y_pred, y_true = [], []
    with torch.no_grad():
        for tbatch in test_loader:
            if torch.cuda.is_available() and device == 'cuda':
                tbatch = {key: val.to(device) for key, val in tbatch.items()}
            outputs = model(tbatch['input_ids'], attention_mask=tbatch['attention_mask'])
            y_pred.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            y_true.extend(tbatch['labels'].cpu().numpy())
    # Compute the metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {acc*100:.2f}%")
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Test F1 Score: {f1*100:.2f}%")
    return acc, f1
    

if __name__ == '__main__':
    # Read the preprocessed data
    print("Load dataset and preprocessing.")
    data_df = pd.read_csv(data_path + args.data_file_name, sep='\t', header=None)
    labels_df = pd.read_csv(labels_path + args.data_file_name, sep='\t', header=None)
    data_df.columns, labels_df.columns = ['Text'], ['Index', 'Split', 'Taxonomy']
    data, labels = data_df['Text'].values.tolist(), labels_df['Taxonomy'].values.tolist()
    # Use small dataset for testing
    NUM_CLASS = len(np.unique(labels))
    # Encode the labels
    label_encoder = LabelEncoder()
    labels_ec = label_encoder.fit_transform(labels)
    # Split the data into train, validation, and test data
    X_train, X_test, y_train, y_test = train_test_split(data, labels_ec, 
                                        test_size=TEST_RATIO, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                        test_size=VAL_RATIO, random_state=seed)

    # Load the tokenizer, pre-trained model, and define the optimizer.
    print("Initialize the tokenizer and pre-trained model.")
    if args.model_name == 'Llama2':
        tokenizer = LlamaTokenizerFast.from_pretrained(model_type_dict[args.model_name],
                                                       use_auth_token=args.access_token)
        # It's crucial to set up pad_token: https://github.com/huggingface/transformers/issues/22312
        tokenizer.pad_token='[PAD]'
        model = LlamaForSequenceClassification.from_pretrained(model_type_dict[args.model_name],
                                                               num_labels=NUM_CLASS,
                                                               use_auth_token=args.access_token)
        args.BATCH_SIZE = 8
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_type_dict[args.model_name])
        model = AutoModelForSequenceClassification.from_pretrained(model_type_dict[args.model_name],
                                                                   num_labels=NUM_CLASS,
                                                                   ignore_mismatched_sizes=is_ignore)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR)
    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.set_device(args.GPU)
        model.to(args.device)
    print(f"Model size: {model.num_parameters()/1000000:.2f} M")

    # Tokenize the input texts
    print("Tokenize the texts and create data loaders.")
    train_tokens = tokenizer(X_train, truncation=True, padding=True, return_tensors='pt')
    val_tokens = tokenizer(X_val, truncation=True, padding=True, return_tensors='pt')    
    test_tokens = tokenizer(X_test, truncation=True, padding=True, return_tensors='pt')

    # Create PyTorch datasets
    train_data = MyData(train_tokens, y_train)
    val_data = MyData(val_tokens, y_val)
    test_data = MyData(test_tokens, y_test)
    # Create DataLoader for the train and test data
    train_loader = DataLoader(train_data, batch_size=args.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.BATCH_SIZE, shuffle=False)

    # Training
    print("Start training:")
    train(model, optimizer, ckpt_dirs, train_loader, val_loader, args.NUM_EPOCHS, args.device)

    # Evaluation
    print("Evaluation:")
    evaluation(model, optimizer, ckpt_path, test_loader, args.device)
    torch.cuda.empty_cache()
    warnings.resetwarnings()
