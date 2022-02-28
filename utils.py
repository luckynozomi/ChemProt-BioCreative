import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch._C import Value

def get_training_split_on_pmid(df, test_size=0.20, random_state=1):

    df = [int(pmid) for pmid in df]
    all_pmids = sorted(list(set(df)))
    ids_train, ids_val = train_test_split(all_pmids, test_size=test_size, random_state=random_state)

    return ids_train, ids_val


def transform_sentence(sentences, chemicals, genes):
    transformed_sents = []
    for sent, chemical, gene in zip(sentences, chemicals, genes):
        _, _, chem_start, chem_end, _ = chemical
        _, _, gene_start, gene_end, _ = gene
        is_chem_first = chem_start < gene_start
        low_start, low_end = (chem_start, chem_end) if chem_start < gene_start else (gene_start, gene_end)
        high_start, high_end = (chem_start, chem_end) if chem_start > gene_start else (gene_start, gene_end)
        tokenized_sent = [sent[0:low_start], sent[low_start:low_end], sent[low_end:high_start], sent[high_start:high_end], sent[high_end:]]
        if is_chem_first:
            tokenized_sent[1] = "chem_name"
            tokenized_sent[3] = "gene_name"
        else:
            tokenized_sent[1] = "gene_name"
            tokenized_sent[3] = "chem_name"
        transformed_sents.append("".join(tokenized_sent))
    return transformed_sents


label_dict = {0: 'AGONIST', 1: 'ANTAGONIST', 2: 'SUBSTRATE', 3: 'AGONIST-INHIBITOR', 4: 'DIRECT-REGULATOR', 
              5: 'INDIRECT-UPREGULATOR', 6: 'SUBSTRATE_PRODUCT-OF', 7: 'NOT', 8: 'ACTIVATOR', 
              9: 'INDIRECT-DOWNREGULATOR', 10: 'INHIBITOR', 11: 'PRODUCT-OF', 12: 'AGONIST-ACTIVATOR', 13: 'PART-OF'}
true_labels_num = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
true_labels_str = set(label_dict.values())
true_labels_str.remove('NOT')


def calculate_f1(true_classes, pred_classes):
    class_true = type(true_classes[0])
    class_pred = type(pred_classes[0])
    if class_true == str and class_pred == str:
        return f1_score(true_classes, pred_classes, labels=true_labels_str, average='micro')
    else:
        return f1_score(true_classes, pred_classes, labels=true_labels_num, average='micro')


def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
