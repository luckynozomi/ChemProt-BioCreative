from operator import pos
from nltk.lm.smoothing import KneserNey
from sentence_transformers import SentenceTransformer, SentencesDataset, losses, models, util
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from sklearn import metrics

from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from utils import transform_sentence, label_dict, get_training_split_on_pmid

train_batch_size = 16
device = 'cpu'

df = pd.read_json("processed_dataset/train_dataset.json", orient="table")
df2 = pd.read_json("processed_dataset/development_dataset.json", orient="table")
df_all = pd.concat([df, df2], ignore_index=True)

ids_train, ids_val = get_training_split_on_pmid(df_all['pmid'].unique())

sentences, chemicals, genes = df_all["sentence"], df_all["chemical"], df_all["gene"]
df_all.loc[:, "sentence"] = transform_sentence(sentences, chemicals, genes)

whole_df = df_all

df_train = whole_df[whole_df['pmid'].isin(ids_train)]
df_val = whole_df[whole_df['pmid'].isin(ids_val)]

df3 = pd.read_json("processed_dataset/test_dataset.json", orient="table")
sentences, chemicals, genes = df3["sentence"], df3["chemical"], df3["gene"]
df3.loc[:, "sentence"] = transform_sentence(sentences, chemicals, genes)
df3 = df3.drop(columns=["all_chemicals", "all_geneYs", "all_geneNs"])

df_test = df3

train_pred_file = 't5data/data_train_predictions_outputs.txt-1208000'
val_pred_file = 't5data/data_val_predictions_outputs.txt-1208000'
test_pred_file = 't5data/data_test_predictions_outputs.txt-1208000'


train_preds = []
with open(train_pred_file, "r") as in_file:
    train_preds = [line.strip() for line in in_file]

val_preds = []
with open(val_pred_file, "r") as in_file:
    val_preds = [line.strip() for line in in_file]

test_preds = []
with open(test_pred_file, "r") as in_file:
    test_preds = [line.strip() for line in in_file]

df_train["pred_label"] = train_preds
df_val["pred_label"] = val_preds
df_test["pred_label"] = test_preds

df_train.to_json("t5_train.json", orient='table')
df_val.to_json("t5_val.json", orient='table')
df_test.to_json("t5_test.json", orient='table')

with open("t5.tsv", "w") as pred_file:
    for _, entry in df_test.iterrows():
        items = [entry["pmid"], entry["pred_label"], "Arg1:{}".format(entry["chemical"][0]), "Arg2:{}".format(entry["gene"][0])]
        string_w = '\t'.join(items)
        pred_file.write(string_w+'\n')