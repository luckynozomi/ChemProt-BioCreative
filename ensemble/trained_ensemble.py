import pandas as pd
import numpy as np
from sklearn.metrics import cluster, f1_score
from utils import label_dict, calculate_f1
import pickle

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


def assert_same_order(df1, df2):
    assert len(df1) == len(df2)
    assert all([a == b for a, b in zip(df1["pmid"], df2["pmid"])]) == True
    assert all([a[0] == b[0] for a, b in zip(df1["chemical"], df2["chemical"])]) == True
    assert all([a[0] == b[0] for a, b in zip(df1["gene"], df2["gene"])]) == True

if __name__ == "__main__":
    str_to_int_labels = {val: key for key, val in label_dict.items()}

    train_bert_probs, train_bert_vecs, df_bert_train = read_bert_files("train")
    _, train_cluster_labels, _, df_sbert_train = read_sbert_files("train")
    assert_same_order(df_bert_train, df_sbert_train)
    max_label = max(train_cluster_labels)
    sbert_train_int_prediction = [str_to_int_labels[label] for label in df_sbert_train["pred_label"]]

    val_bert_probs, val_bert_vecs, df_bert_val = read_bert_files("val")
    _, val_cluster_labels, _, df_sbert_val = read_sbert_files("val")
    assert_same_order(df_bert_val, df_sbert_val)
    sbert_val_int_prediction = [str_to_int_labels[label] for label in df_sbert_val["pred_label"]]

    # using the smaller set of dev data because probabaly some info leak when training
    # og_val = pd.read_json("processed_dataset/development_dataset.json", orient="table")
    # og_ids = og_val["pmid"].unique()
    # subset_idx = df_sbert_val["pmid"].isin(og_ids)
    # df_bert_val = df_bert_val[subset_idx]
    # df_sbert_val = df_sbert_val[subset_idx]
    # val_bert_probs = val_bert_probs[subset_idx, :]
    # val_bert_vecs = val_bert_vecs[subset_idx, :]
    # val_cluster_labels = val_cluster_labels[subset_idx]
    # sbert_val_int_prediction = [str_to_int_labels[label] for label in df_sbert_val["pred_label"]]

    test_bert_vecs, test_bert_probs, df_bert_test = read_bert_files("test")
    _, test_cluster_labels, _, df_sbert_test = read_sbert_files("test")
    assert_same_order(df_bert_test, df_sbert_test)
    sbert_test_int_prediction = [str_to_int_labels[label] for label in df_sbert_test["pred_label"]]


    features_train = np.column_stack([train_bert_vecs, train_bert_probs, train_cluster_labels, sbert_train_int_prediction])
    features_val = np.column_stack([val_bert_vecs, val_bert_probs, val_cluster_labels, sbert_val_int_prediction])
    features_test = np.column_stack([test_bert_vecs, test_bert_probs, test_cluster_labels, sbert_test_int_prediction])


    def get_response(sbert_prediction, bert_prediction, true_label):
        assert len(sbert_prediction) == len(bert_prediction)
        assert len(bert_prediction) == len(true_label)
        ret = []
        for sbert, bert, true in zip(sbert_prediction, bert_prediction, true_label):
            if sbert == true and bert == true:
                ret.append(0)  # 11497
            elif sbert == true and bert != true:
                ret.append(1)  # 545
            elif sbert != true and bert == true:
                ret.append(2)  # 231
            else:
                ret.append(3)  # 90
        return ret

    response_train = get_response(df_sbert_train["pred_label"], df_bert_train["pred_label"], df_sbert_train["relation type"])
    response_val = get_response(df_sbert_val["pred_label"], df_bert_val["pred_label"], df_sbert_val["relation type"])


    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

    CLF = {
        'xgb': {'cv_param': {'learning_rate': [.01, .05, .1, .5],
                                    'max_depth': [3,5,7],
                                    'min_child_weight': [1,2,3]},
                        'estimator': xgb.XGBClassifier(n_estimators=100, nthread=-1, subsample=.9, objective='multi:softmax', eval_metric='mlogloss', tree_method='gpu_hist', gpu_id=0)
                        },
        'lr': {'cv_param': {'C': [.01, .05, .1, .5, 1.0, 5.0, 10.0],
                            'penalty': ['l2']},
                'estimator': LogisticRegression()
                },
        'et': {'cv_param': {'criterion': ['gini','entropy'],
                            'max_depth': [3,5,7,None],
                            'n_estimators': [10,20,30,50,100]}, 
                'estimator': ExtraTreesClassifier( n_jobs=-1)
                }, 
        'rf': {'cv_param': {'criterion': ['gini','entropy'],
                            'max_depth': [3,5,7,None],
                            'n_estimators': [10,20,30,50,100]},
                'estimator':RandomForestClassifier(n_jobs=-1)
                }
    }

    from sklearn.model_selection import RandomizedSearchCV

    predicted_probs = np.zeros([len(CLF), len(features_val), 4]) # 4 because 4 classes
    for clf_idx, clf_name in enumerate(CLF.keys()):
        clf = CLF[clf_name]['estimator']
        clf_param = CLF[clf_name]['cv_param']
        random_search = RandomizedSearchCV(clf, clf_param, n_iter=10, cv=5, scoring='accuracy')
        random_search.fit(features_train, response_train)
        clf_aftersearch = random_search.best_estimator_
        pickle.dump(clf_aftersearch, open("Model{}_{}_old.pickle".format(clf_idx, clf_name), "wb"))
        predict_proba = clf_aftersearch.predict_proba(features_val)
        predicted_probs[clf_idx, :] = predict_proba
    average_proba = np.mean(predicted_probs, axis=0)
    predicted_class = np.argmax(average_proba, axis=1)

    test_classes = predicted_class

    predictions = []
    real_classes = []
    all_classes = set()
    for idx, test_class in enumerate(test_classes):
        bert_prediction = df_bert_val["pred_label"].iloc[idx]
        sbert_prediction = df_sbert_val["pred_label"].iloc[idx]
        if test_class == 1:
            predictions.append(sbert_prediction)
        else:
            predictions.append(bert_prediction)
        real_class = df_sbert_val["relation type"].iloc[idx]
        real_classes.append(real_class)
        all_classes.add(real_class)
        all_classes.add(sbert_prediction)
        all_classes.add(bert_prediction)

    all_classes.remove("NOT")
    from sklearn.metrics import f1_score
    f1 = f1_score(real_classes, predictions, labels=list(all_classes), average="micro")
    print(f1)
    # 0.7538997034936187