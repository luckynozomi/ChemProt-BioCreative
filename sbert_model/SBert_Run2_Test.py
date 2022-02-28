from sentence_transformers import SentenceTransformer, SentencesDataset, losses, models, util
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import LabelAccuracyEvaluator
import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from utils import transform_sentence, get_training_split_on_pmid, label_dict

train_batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df1 = pd.read_json("processed_dataset/train_dataset.json", orient="table")
df2 = pd.read_json("processed_dataset/development_dataset.json", orient="table")
df3 = pd.read_json("processed_dataset/test_dataset.json", orient="table")
df_all = pd.concat([df1, df2], ignore_index=True)

ids_train, ids_val = get_training_split_on_pmid(df_all['pmid'].unique())

sentences, chemicals, genes = df_all["sentence"], df_all["chemical"], df_all["gene"]
df_all.loc[:, "sentence"] = transform_sentence(sentences, chemicals, genes)

df_all = df_all.drop(columns=["all_chemicals", "all_geneYs", "all_geneNs"])

df_train = df_all[df_all['pmid'].isin(ids_train)]
df_val = df_all[df_all['pmid'].isin(ids_val)]
df_test = df3

print("Sample size: train: {}, val: {}".format(len(df_train), len(df_val)))

relation_to_label_dict = {val: key for key, val in label_dict.items()}

relation_train = [relation_to_label_dict[relation] for relation in df_train["relation type"]]
relation_val = [relation_to_label_dict[relation] for relation in df_val["relation type"]]

df_train["label"] = relation_train
df_val["label"] = relation_val

label_list = set(df_train['label'])

train_examples = []
train_sentences = []
for _, example in df_train.iterrows():
    sentence = example["sentence"]
    relation_type = example["relation type"]
    label = relation_to_label_dict[relation_type]
    train_examples.append(InputExample(texts=[sentence], label=label))
    train_sentences.append(sentence)

val_examples = []
val_sentences = []
for _, example in df_val.iterrows():
    sentence = example["sentence"]
    relation_type = example["relation type"]
    label = relation_to_label_dict[relation_type]
    val_examples.append(InputExample(texts=[sentence], label=label))
    val_sentences.append(sentence)

test_examples = []
test_sentences = []
for _, example in df_test.iterrows():
    sentence = example["sentence"]
    relation_type = example["relation type"]
    label = relation_to_label_dict[relation_type]
    test_examples.append(InputExample(texts=[sentence], label=label))
    test_sentences.append(sentence)

model_name = "auto_sbert_microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_256_5.0_10_2e-05"

model = SentenceTransformer(model_name)

train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.BatchHardTripletLoss(model=model)

test_dataset = SentencesDataset(test_examples, model)
test_dataloader = DataLoader(test_dataset, batch_size=1)
evaluator = LabelAccuracyEvaluator(dataloader=test_dataloader)

train_vecs = model.encode(train_sentences, convert_to_numpy=True, show_progress_bar=True)
train_labels = [train_example.label for train_example in train_examples]
import numpy as np
np.save("train_vectors.npy", train_vecs)

import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=1, prediction_data=True)
clusterer.fit(train_vecs)

import pickle
pickle.dump(clusterer, open("clusterer.pickle", "wb"))

np.save("train_cluster_labels.npy", clusterer.labels_)

print(max(clusterer.labels_))
print(sum(clusterer.labels_))

print("Max cluster ID", max(clusterer.labels_))

from sklearn.neighbors import KNeighborsClassifier
N_NEIGHBORS = 1000  # 3 - 1000 works
# do clustering first
# see which cluster has the most error and look into that cluster
neigh = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
neigh.fit(train_vecs, train_labels)

pickle.dump(neigh, open("neigh.pickle", "wb"))

# training set
train_vecs = model.encode(train_sentences, convert_to_numpy=True, show_progress_bar=True)
train_labels = neigh.predict(train_vecs)
true_train_labels = [train_example.label for train_example in train_examples]

out_file = "cl_predictions_train_{}_{}.tsv".format(model_name.replace("/", "_"), N_NEIGHBORS)
ret_train = copy.deepcopy(df_train)
pred_classes = []
out_f = open(out_file, "w")
for idx in range(len(true_train_labels)):
    true_label = true_train_labels[idx]
    pred_label = train_labels[idx]
    true_class, pred_class = label_dict[true_label], label_dict[pred_label]
    pred_classes.append(pred_class)
    this_instance = df_train.iloc[idx, :].squeeze()
    pmid = this_instance["pmid"]
    arg1 = "Arg1:{}".format(this_instance["chemical"][0])
    arg2 = "Arg2:{}".format(this_instance["gene"][0])
    out_f.write(pmid+'\t'+pred_class+'\t'+arg1+'\t'+arg2+'\n')
out_f.close()
ret_train["pred_label"] = pred_classes
ret_train.to_json("sbert_train_df.json", orient="table")


# validation set
val_vecs = model.encode(val_sentences, convert_to_numpy=True, show_progress_bar=True)
val_labels = neigh.predict(val_vecs)
true_val_labels = [val_example.label for val_example in val_examples]

val_cluster_labels, val_cluster_strengths = hdbscan.approximate_predict(clusterer, val_vecs)

np.save("val_vectors.npy", val_vecs)
np.save("val_cluster_labels.npy", val_cluster_labels)
np.save("val_cluster_strengthes.npy", val_cluster_strengths)

out_file = "cl_predictions_val_{}_{}.tsv".format(model_name.replace("/", "_"), N_NEIGHBORS)
out_f = open(out_file, "w")
ret_val = copy.deepcopy(df_val)
pred_classes = []
for idx in range(len(true_val_labels)):
    true_label = true_val_labels[idx]
    pred_label = val_labels[idx]
    true_class, pred_class = label_dict[true_label], label_dict[pred_label]
    pred_classes.append(pred_class)
    this_instance = df_val.iloc[idx, :].squeeze()
    pmid = this_instance["pmid"]
    arg1 = "Arg1:{}".format(this_instance["chemical"][0])
    arg2 = "Arg2:{}".format(this_instance["gene"][0])
    out_f.write(pmid+'\t'+pred_class+'\t'+arg1+'\t'+arg2+'\n')
out_f.close()
ret_val["pred_label"] = pred_classes
ret_val.to_json("sbert_val_df.json", orient='table')



test_vecs = model.encode(test_sentences, convert_to_numpy=True, show_progress_bar=True)
test_labels = neigh.predict(test_vecs)
true_test_labels = [test_example.label for test_example in test_examples]

test_cluster_labels, cluster_strengths = hdbscan.approximate_predict(clusterer, test_vecs)

np.save("test_vectors.npy", test_vecs)
np.save("test_cluster_labels.npy", test_cluster_labels)
np.save("test_cluster_strengthes.npy", cluster_strengths)

ret_test = copy.deepcopy(df_test)
out_file = "cl_predictions_test_{}_{}.tsv".format(model_name.replace("/", "_"), N_NEIGHBORS)
out_f = open(out_file, "w")
pred_classes = []
for idx in range(len(true_test_labels)):
    true_label = true_test_labels[idx]
    pred_label = test_labels[idx]
    true_class, pred_class = label_dict[true_label], label_dict[pred_label]
    pred_classes.append(pred_class)
    this_instance = df_test.iloc[idx, :].squeeze()
    pmid = this_instance["pmid"]
    arg1 = "Arg1:{}".format(this_instance["chemical"][0])
    arg2 = "Arg2:{}".format(this_instance["gene"][0])
    out_f.write(pmid+'\t'+pred_class+'\t'+arg1+'\t'+arg2+'\n')
out_f.close()
ret_test["pred_label"] = pred_classes
ret_test.to_json("sbert_test_df.json", orient="table")
