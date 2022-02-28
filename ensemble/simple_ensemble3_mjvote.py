import pandas as pd
import numpy as np
from sklearn.metrics import cluster, f1_score


def read_bert_files(prefix):
    probs = np.load("{}_probs.npy".format(prefix))
    features = np.load("{}_vecs.npy".format(prefix))
    dataframe = pd.read_json("bert_{}_df.json".format(prefix), orient="table")
    return probs, features, dataframe


def read_sbert_files(prefix):
    vectors = np.load("{}_vectors.npy".format(prefix))
    cluster_labels = np.load("{}_cluster_labels.npy".format(prefix))
    cluster_strengths = np.load("{}_cluster_strengthes.npy".format(prefix)) if prefix != "train" else None
    dataframe = pd.read_json("sbert_{}_df.json".format(prefix), orient="table")
    return vectors, cluster_labels, cluster_strengths, dataframe


def read_t5_files(prefix):
    dataframe = pd.read_json("t5_{}.json".format(prefix), orient='table')
    return dataframe


def assert_same_order(df1, df2):
    assert len(df1) == len(df2)
    assert all([a == b for a, b in zip(df1["pmid"], df2["pmid"])]) == True
    assert all([a[0] == b[0] for a, b in zip(df1["chemical"], df2["chemical"])]) == True
    assert all([a[0] == b[0] for a, b in zip(df1["gene"], df2["gene"])]) == True


def mjvote(sbert_pred, bert_pred, t5pred):
    if sbert_pred == bert_pred:
        return sbert_pred
    elif bert_pred == t5_pred:
        return bert_pred
    elif sbert_pred == t5_pred:
        return sbert_pred
    else:
        return t5pred

if __name__ == "__main__":
    _, _, df_bert_train = read_bert_files("train")
    _, cluster_labels, _, df_sbert_train = read_sbert_files("train")
    df_t5_train = read_t5_files("train")
    assert_same_order(df_bert_train, df_sbert_train)
    df_sbert_train["cluster_label"] = cluster_labels
    max_label = max(cluster_labels)

    _, _, df_bert_val = read_bert_files("val")
    _, cluster_labels, _, df_sbert_val = read_sbert_files("val")
    df_t5_val = read_t5_files("val")

    assert_same_order(df_bert_val, df_sbert_val)
    df_sbert_val["cluster_label"] = cluster_labels

    # using the smaller set of dev data because probabaly some info leak when training
    # og_val = pd.read_json("processed_dataset/development_dataset.json", orient="table")
    # og_ids = og_val["pmid"].unique()
    # subset_idx = df_sbert_val["pmid"].isin(og_ids)
    # df_bert_val = df_bert_val[subset_idx]
    # df_sbert_val = df_sbert_val[subset_idx]

    _, _, df_bert_test = read_bert_files("test")
    _, cluster_labels, _, df_sbert_test = read_sbert_files("test")
    df_t5_test = read_t5_files("test")

    assert_same_order(df_bert_test, df_sbert_test)
    df_sbert_test["cluster_label"] = cluster_labels


    use_model_dict = {}
    for cluster_id in range(-1, max_label+1):
        idxes = list(df_sbert_train["cluster_label"] == cluster_id)
        sub_bert = df_bert_train[idxes]
        sub_sbert = df_sbert_train[idxes]
        sub_t5 = df_t5_train[idxes]
        bert_preds = sub_bert["pred_label"]
        sbert_preds = sub_sbert["pred_label"]
        t5_preds = sub_t5["pred_label"]
        real_labels = sub_sbert["relation type"]

        all_classes = set(bert_preds)
        all_classes = all_classes.union(set(sbert_preds))
        all_classes = all_classes.union(set(real_labels))
        all_classes = all_classes.union(set(t5_preds))
        if 'NOT' in all_classes:
            all_classes.remove("NOT")
        # bert_f1 = f1_score(real_labels, bert_preds, labels=list(all_classes), average='micro')
        # sbert_f1 = f1_score(real_labels, sbert_preds, labels=list(all_classes), average='micro')
        # t5_f1 = f1_score(real_labels, t5_preds, labels=list(all_classes), average='micro')
        from sklearn.metrics import accuracy_score
        bert_f1 = accuracy_score(real_labels, bert_preds)
        sbert_f1 = accuracy_score(real_labels, sbert_preds)
        t5_f1 = accuracy_score(real_labels, t5_preds)
        if t5_f1 > sbert_f1 and t5_f1 > bert_f1:
            use_model_dict[cluster_id] = 2  # T5
        elif sbert_f1 > bert_f1:
            use_model_dict[cluster_id] = 1  # SBERT
        else:
            use_model_dict[cluster_id] = 0  # BERT
        print(bert_f1, sbert_f1, t5_f1)
        print(set(sbert_preds))

    true_all_classes = set(df_sbert_val["relation type"])
    true_all_classes = true_all_classes.union(set(df_bert_val["pred_label"]))
    true_all_classes.remove("NOT")
    true_all_classes = list(true_all_classes)

    preds = []
    reals = []
    for val_index in range(len(df_sbert_val)):
        bert_entry = df_bert_val.iloc[val_index].squeeze()
        sbert_entry = df_sbert_val.iloc[val_index].squeeze()
        t5_entry = df_t5_val.iloc[val_index].squeeze()
        real_label = sbert_entry["relation type"]
        cluster_id = sbert_entry["cluster_label"]
        sbert_pred = sbert_entry["pred_label"]
        bert_pred = bert_entry["pred_label"]
        t5_pred = t5_entry["pred_label"]
        pred = mjvote(sbert_pred, bert_pred, t5_pred)
        preds.append(pred)
        reals.append(real_label)
    f1s = f1_score(reals, preds, labels=true_all_classes, average='micro')
    from sklearn.metrics import confusion_matrix
    all_LABS = true_all_classes + ['AGONIST-INHIBITOR']
    all_LABS.sort()
    all_LABS = all_LABS + ["NOT"]
    print(all_LABS)
    print(confusion_matrix(reals, preds, labels=all_LABS))
    print(f1s)
    print(use_model_dict)

    bert_f1 = f1_score(df_sbert_val["relation type"], df_bert_val["pred_label"], labels=true_all_classes, average="micro")
    sbert_f1 = f1_score(df_sbert_val["relation type"], df_sbert_val["pred_label"], labels=true_all_classes, average="micro")
    t5_f1 = f1_score(df_t5_val["relation type"], df_t5_val["pred_label"], labels=true_all_classes, average="micro")
    print(bert_f1, sbert_f1, t5_f1)
    # 0.7477322873253248 0.7646237787082858
    # {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 1, 12: 1}
    # 0.7335452725774166 0.7333594873806721 0.7476466795615733
    prediction_file = open("Simple_Average_MJVote.tsv", "w")
    for test_idx in range(len(df_sbert_test)):  # CHANGE IT
        bert_entry = df_bert_test.iloc[test_idx].squeeze()  # CHANGE IT
        sbert_entry = df_sbert_test.iloc[test_idx].squeeze()  # CHANGE IT
        t5_entry = df_t5_test.iloc[test_idx].squeeze()
        pmid = sbert_entry["pmid"]
        arg1 = sbert_entry["chemical"][0]
        arg2 = sbert_entry["gene"][0]
        cluster_id = sbert_entry["cluster_label"]
        sbert_pred = sbert_entry["pred_label"]
        bert_pred = bert_entry["pred_label"]
        t5_pred = t5_entry["pred_label"]
        # print(sbert_pred, bert_pred, t5_pred)
        pred = mjvote(sbert_pred, bert_pred, t5_pred)
        if pred != "NOT":
            prediction_file.write(pmid+'\t'+pred+'\t'+"Arg1:{}".format(arg1)+'\t'+"Arg2:{}".format(arg2)+'\n')
    prediction_file.close()