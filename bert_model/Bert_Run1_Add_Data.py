#!/usr/bin/env python
# coding: utf-8

from random import randrange

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import transformers
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import defaultdict
import numpy as np

from utils import get_training_split_on_pmid, transform_sentence, label_dict
from BertModel import data_loader, RelationClassifier, main_train_fn

RANDOM_SEED = 40
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transformers.logging.set_verbosity_error()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



df1 = pd.read_json("processed_dataset/train_dataset.json", orient="table")
df2 = pd.read_json("processed_dataset/development_dataset.json", orient="table")
df1 = df1.rename(columns={"relation type": 'label'})
df2 = df2.rename(columns={"relation type": 'label'})
for label in label_dict:
    df1['label'] = df1['label'].replace([label_dict[label]],label)
    df2['label'] = df2['label'].replace([label_dict[label]],label)


train_sentences, train_chemicals, train_genes = df1["sentence"], df1["chemical"], df1["gene"]
dev_sentences, dev_chemicals, dev_genes = df2["sentence"], df2["chemical"], df2["gene"]
df1.loc[:,"sentence"] = transform_sentence(train_sentences, train_chemicals, train_genes)
df2.loc[:,"sentence"] = transform_sentence(dev_sentences, dev_chemicals, dev_genes)

df = df1.append(df2)
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
LEARNING_RATE = 7e-7
SAVE_PATH = 'best_model_Pubmenbert_full_1.bin'

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False) 
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_data) * EPOCHS
    )
loss_fn = nn.CrossEntropyLoss().to(device)  


history = defaultdict(list)
best_accuracy = 0

main_train_fn(model, train_data, val_data, EPOCHS, loss_fn, optimizer, device, scheduler, SAVE_PATH)


def generate_augmented_data(data_df, shortest_path_df_dir, device, tokenizer, MAX_LEN, BATCH_SIZE):
    data_df_loader = data_loader(data_df, tokenizer, MAX_LEN, BATCH_SIZE)
    y_sentence_texts, y_pred, y_pred_probs, y_test, _ = model.get_predictions(data_df_loader, device)
    pred = y_pred.tolist()
    indexes = []
    for i in range(0,len(y_pred)):
        if pred[i] != data_df.label[i]:
            indexes.append(i)

    lines = [line.rstrip() for line in open(shortest_path_df_dir)]
    shortest_path_df = pd.DataFrame(columns = ['id','sp'])
    for line in lines:
        l = line.split('\t')
        shortest_path_df = shortest_path_df.append({'id' : l[0], 'sp' : ' '.join(l[1:])}, ignore_index = True)

    df_new = data_df.iloc[0:0,:]
    for i in indexes:
        sent = data_df.sentence[i]
        sp = shortest_path_df['sp'][i]
        s = sent.split(" ")
        p = sp.split(" ")
        k = randrange(len(s))
        while s[k] in p:
            k = randrange(len(s))
        del s[k]
        sentence = " ".join(s)
        df_new = df_new.append({'pmid' : data_df.pmid[i], 'chemical' : data_df.chemical[i], 'gene' : data_df.gene[i],
                            'all_chemicals' : data_df.all_chemicals[i], 'all_geneNs' : data_df.all_geneNs[i], 
                                'label' : data_df.label[i], 'sentence' : sentence}, ignore_index = True)
        
    for label in label_dict:
        df_new['label'] = df_new['label'].replace(label, label_dict[label])
    
    return df_new


df_train_new = generate_augmented_data(df1, 'shortest_path.tsv', device, tokenizer, MAX_LEN, BATCH_SIZE)
df_dev_new = generate_augmented_data(df2, 'shortest_path_dev.tsv', device, tokenizer, MAX_LEN, BATCH_SIZE)
df_train_new.to_csv('train_add_Pubfull.csv')
df_dev_new.to_csv('dev_add_Pubfull.csv')


def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
