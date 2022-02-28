#!/usr/bin/env python
# coding: utf-8

# In[1]:

import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random 
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import Adafactor,AdamWeightDecay
from collections import defaultdict
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
import numpy as np
from utils import label_dict, transform_sentence, get_training_split_on_pmid
from BertModel import data_loader, RelationClassifier
import numpy as np
import csv

RANDOM_SEED = 40
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transformers.logging.set_verbosity_error()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


df1 = pd.read_json("processed_dataset/train_dataset.json", orient="table")
df2 = pd.read_json("processed_dataset/development_dataset.json", orient="table")
df3 = pd.read_json("processed_dataset/test_dataset.json", orient="table")
df1 = df1.rename(columns={"relation type": 'label'})
df2 = df2.rename(columns={"relation type": 'label'})
df3 = df3.rename(columns={"relation type": 'label'})
for label in label_dict:
    df1['label'] = df1['label'].replace([label_dict[label]],label)
    df2['label'] = df2['label'].replace([label_dict[label]],label)
    df3['label'] = df3['label'].replace([label_dict[label]],label)


train_sentences, train_chemicals, train_genes = df1["sentence"], df1["chemical"], df1["gene"]
dev_sentences, dev_chemicals, dev_genes = df2["sentence"], df2["chemical"], df2["gene"]
test_sentences, test_chemicals, test_genes = df3["sentence"], df3["chemical"], df3["gene"]

df1.loc[:,"sentence"] = transform_sentence(train_sentences, train_chemicals, train_genes)
df2.loc[:,"sentence"] = transform_sentence(dev_sentences, dev_chemicals, dev_genes)
df3.loc[:,"sentence"] = transform_sentence(test_sentences, test_chemicals, test_genes)

df = df1.append(df2)
ids_train, ids_val = get_training_split_on_pmid(df)
df_train = df[df['pmid'].isin(ids_train)]
df_val = df[df['pmid'].isin(ids_val)]
df_test = df3



# Model Parameters
PRE_TRAINED_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
MAX_LEN = 64
BATCH_SIZE = 32
BEST_MODEL = "best_model_Pubmenbert_full_2_6e-06.bin"

n_classes = len(label_dict)
model = RelationClassifier(n_classes, bert_pretrain_path=PRE_TRAINED_MODEL_NAME)
model.load_state_dict(torch.load(BEST_MODEL))
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
train_data = data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data = data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data = data_loader(df_test, tokenizer, None, 1)


def gather_features_responses(dataframe, tokenizer, MAX_LEN, BATCH_SIZE, prefix, device):

    dataset = data_loader(dataframe, tokenizer, MAX_LEN, BATCH_SIZE)

    y_sentence_texts, y_pred, y_pred_probs, y_test, vecs = model.get_predictions(dataset, device)
    np.save("{}_probs".format(prefix), y_pred_probs.numpy())
    np.save("{}_vecs".format(prefix), vecs.numpy())

    y_pred_name = []
    for i in y_pred:
        y_pred_name.append(label_dict[int(i)])
    with open('output_bert_{}.tsv'.format(prefix), 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for i in range(0,len(y_pred_name)):
            entry = dataframe.iloc[i,:]
            tsv_writer.writerow([entry.pmid, y_pred_name[i], 'Arg1:'+entry.chemical[0], 'Arg2:'+entry.gene[0]])
    new_df = copy.deepcopy(dataframe)
    new_df["pred_label"] = y_pred_name
    new_df.to_json("bert_{}_df.json".format(prefix), orient='table')


gather_features_responses(df_train, tokenizer, MAX_LEN, BATCH_SIZE, "train", device)
gather_features_responses(df_val, tokenizer, None, 1, "val", device)
gather_features_responses(df_test, tokenizer, None, 1, "test", device)
