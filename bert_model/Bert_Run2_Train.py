#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import torch
from torch import nn
import transformers
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np

from utils import get_training_split_on_pmid, transform_sentence, label_dict
from BertModel import data_loader, RelationClassifier, main_train_fn

RANDOM_SEED = 40
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformers.logging.set_verbosity_error()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


df1 = pd.read_json("processed_dataset/train_dataset.json", orient="table")
df2 = pd.read_json("processed_dataset/development_dataset.json", orient="table")
df1 = df1.rename(columns={"relation type": 'label'})
df2 = df2.rename(columns={"relation type": 'label'})

train_sentences, train_chemicals, train_genes = df1["sentence"], df1["chemical"], df1["gene"]
dev_sentences, dev_chemicals, dev_genes = df2["sentence"], df2["chemical"], df2["gene"]
df1.loc[:,"sentence"] = transform_sentence(train_sentences, train_chemicals, train_genes)
df2.loc[:,"sentence"] = transform_sentence(dev_sentences, dev_chemicals, dev_genes)


df_train_new = pd.read_csv('train_add_Pubfull.csv', index_col=0)
df1_add = df1.append(df_train_new)
df_dev_new = pd.read_csv('dev_add_Pubfull.csv', index_col=0)
df2_add = df2.append(df_dev_new)


for label in label_dict:
    df1_add['label'] = df1_add['label'].replace([label_dict[label]],label)
    df2_add['label'] = df2_add['label'].replace([label_dict[label]],label)


df = df1_add.append(df2_add)
ids_train, ids_val = get_training_split_on_pmid(df)
df_train = df[df['pmid'].isin(ids_train)]
df_val = df[df['pmid'].isin(ids_val)]


# Model Parameters
PRE_TRAINED_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
MAX_LEN =64
BATCH_SIZE = 32

tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
train_data = data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data = data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)

n_classes = len(label_dict)
model = RelationClassifier(n_classes, bert_pretrain_path=PRE_TRAINED_MODEL_NAME)
model = model.to(device)

# Training Parameters
EPOCHS = 10
LEARNING_RATE = 7e-6
SAVE_PATH = 'best_model_Pubmenbert_full_2.bin'

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False) #7e-6:0.752，2e-6:0.730
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_data) * EPOCHS
    )
loss_fn = nn.CrossEntropyLoss().to(device)

main_train_fn(model, train_data, val_data, EPOCHS, loss_fn, optimizer, device, scheduler, SAVE_PATH)
