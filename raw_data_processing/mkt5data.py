import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import transform_sentence, get_training_split_on_pmid

train_batch_size = 16
device = 'cpu'

df = pd.read_json("processed_dataset/train_dataset.json", orient="table")
df2 = pd.read_json("processed_dataset/development_dataset.json", orient="table")
df_all = pd.concat([df, df2], ignore_index=True)

ids_train, ids_val = get_training_split_on_pmid(list(df_all['pmid'].unique()))

sentences, chemicals, genes = df_all["sentence"], df_all["chemical"], df_all["gene"]
df_all.loc[:, "sentence"] = transform_sentence(sentences, chemicals, genes)

df_all = df_all.drop(columns=["all_chemicals", "all_geneYs", "all_geneNs"])

whole_df = df_all
# Use following if add data after bert run
# df_add1 = pd.read_csv("train_add_Pubfull.csv", index_col=0)
# df_add2 = pd.read_csv("dev_add_Pubfull.csv", index_col=0)
# df_add = pd.concat([df_add1, df_add2], ignore_index=True)
# df_add = df_add.rename(columns={"label": "relation type"})
# df_add = df_add.drop(columns=["all_chemicals", "all_geneYs", "all_geneNs"])
# df_add = df_add.astype({"pmid": "str"})
# whole_df = pd.concat([df_all, df_add], ignore_index=True)

df_train = whole_df[whole_df['pmid'].isin(ids_train)]
df_val = whole_df[whole_df['pmid'].isin(ids_val)]

df3 = pd.read_json("processed_dataset/test_dataset.json", orient="table")
sentences, chemicals, genes = df3["sentence"], df3["chemical"], df3["gene"]
df3.loc[:, "sentence"] = transform_sentence(sentences, chemicals, genes)
df3 = df3.drop(columns=["all_chemicals", "all_geneYs", "all_geneNs"])

df_test = df3

dfs = [df_train, df_val, df_test]
names = ["train", "val", "test"]
for df, name in zip(dfs, names):
    sentences = df["sentence"]
    interactions  = df["relation type"]
    with open('t5data/{}.tsv'.format(name), "w") as out_file:
        for sent, interaction in zip(sentences, interactions):
            out_file.write(sent+'\t'+interaction+'\n')

    if name != "train.tsv":
        new_sents = ["chemprot_re: " + sent for sent in sentences]
        with open("t5data/{}_predictions.txt".format(name), "w") as out_file:
            out_file.write('\n'.join(new_sents))
        
        with open("t5data/{}_labels.txt".format(name), "w") as out_file:
            out_file.write('\n'.join(interactions))


df = pd.read_json("processed_dataset/train_dataset.json", orient="table")
df2 = pd.read_json("processed_dataset/development_dataset.json", orient="table")
df_all = pd.concat([df, df2], ignore_index=True)

ids_train, ids_val = train_test_split(df_all['pmid'].unique(), test_size=0.2, random_state=42)

sentences, chemicals, genes = df_all["sentence"], df_all["chemical"], df_all["gene"]
df_all.loc[:, "sentence"] = transform_sentence(sentences, chemicals, genes)

df_all = df_all.drop(columns=["all_chemicals", "all_geneYs", "all_geneNs"])

whole_df = df_all

df_train = whole_df[whole_df['pmid'].isin(ids_train)]
df_val = whole_df[whole_df['pmid'].isin(ids_val)]

os.makedirs("processed_dataset/t5_data")

sentences = df_train["sentence"]
interactions  = df_train["relation type"]
with open('processed_dataset/t5_data/train.tsv', "w") as out_file:
    for sent, interaction in zip(sentences, interactions):
        out_file.write(sent+'\t'+interaction+'\n')

new_sents = ["chemprot_re: " + sent for sent in sentences]
with open("processed_dataset/t5_data/train_predictions.txt", "w") as out_file:
    out_file.write('\n'.join(new_sents))


sentences = df_val["sentence"]
interactions  = df_val["relation type"]
with open('processed_dataset/t5_data/val.tsv', "w") as out_file:
    for sent, interaction in zip(sentences, interactions):
        out_file.write(sent+'\t'+interaction+'\n')

new_sents = ["chemprot_re: " + sent for sent in sentences]
with open("processed_dataset/t5_data/val_predictions.txt", "w") as out_file:
    out_file.write('\n'.join(new_sents))
