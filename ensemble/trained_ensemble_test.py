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

    test_bert_vecs, test_bert_probs, df_bert_test = read_bert_files("test")  # CHANGE IT
    _, test_cluster_labels, _, df_sbert_test = read_sbert_files("test")  #CHANGE IT
    assert_same_order(df_bert_test, df_sbert_test)
    sbert_test_int_prediction = [str_to_int_labels[label] for label in df_sbert_test["pred_label"]]
    
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

    predicted_probs = np.zeros([len(CLF), len(features_test), 4]) # 4 because 4 classes
    for clf_idx, clf_name in enumerate(CLF.keys()):
        clf_aftersearch = pickle.load(open("Model{}_{}_old.pickle".format(clf_idx, clf_name), "rb"))
        predict_proba = clf_aftersearch.predict_proba(features_test)
        predicted_probs[clf_idx, :] = predict_proba
    average_proba = np.mean(predicted_probs, axis=0)
    predicted_class = np.argmax(average_proba, axis=1)

    test_classes = predicted_class
    prediction_file = open("Trained_Average.tsv", "w")
    for idx, test_class in enumerate(test_classes):
        bert_prediction = df_bert_test["pred_label"].iloc[idx]
        sbert_prediction = df_sbert_test["pred_label"].iloc[idx]
        prediction = sbert_prediction if test_class == 1 else bert_prediction

        bert_entry = df_bert_test.iloc[idx].squeeze()
        sbert_entry = df_sbert_test.iloc[idx].squeeze()
        pmid = sbert_entry["pmid"]
        arg1 = sbert_entry["chemical"][0]
        arg2 = sbert_entry["gene"][0]

        if prediction != "NOT":
            prediction_file.write(pmid+'\t'+prediction+'\t'+"Arg1:{}".format(arg1)+'\t'+"Arg2:{}".format(arg2)+'\n')
    prediction_file.close()
    # avg: 0.751
    # xgb: 0.745
    # lr: 0.747
    # et: 0.749
    # rf: 0.749