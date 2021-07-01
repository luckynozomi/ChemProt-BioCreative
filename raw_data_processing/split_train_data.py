import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Run make_dataset.py first
df = pd.read_json("train_dataset.json", orient="table")

ids_train, ids_test = train_test_split(df['pmid'].unique(), test_size=0.20, random_state=1)
ids_val, ids_test = train_test_split(ids_test, test_size=0.5, random_state=1)

# ids_train = map(int, ids_train)
# ids_val = map(int, ids_val)
# ids_test = map(int, ids_test)

df_train = df[df['pmid'].isin(ids_train)]
df_val = df[df['pmid'].isin(ids_val)]
df_test = df[df['pmid'].isin(ids_test)]

if not os.path.isdir("split_dataframe"):
    os.mkdir("split_dataframe") 

df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

df_train.to_json("split_dataframe/train_dataset.json", orient="table", indent=4)
df_val.to_json("split_dataframe/val_dataset.json", orient="table", indent=4)
df_test.to_json("split_dataframe/test_dataset.json", orient="table", indent=4)

if not os.path.isdir("split_original"):
    os.mkdir("split_original")

import csv

abstracts = pd.read_csv("train_data/drugprot_training_abstracs.tsv", sep='\t', header=None, names=["pmid", "title", "abstract"], dtype={"pmid": str})
train_abstracts = abstracts[abstracts["pmid"].isin(ids_train)]
val_abstracts = abstracts[abstracts["pmid"].isin(ids_val)]
test_abstracts = abstracts[abstracts["pmid"].isin(ids_test)]
train_abstracts.to_csv("split_original/drugprot_training_abstracts.tsv", index=False, sep='\t', header=False, quoting=csv.QUOTE_NONE)
val_abstracts.to_csv("split_original/drugprot_development_abstracts.tsv", index=False, sep='\t', header=False, quoting=csv.QUOTE_NONE)
test_abstracts.to_csv("split_original/drugprot_test_abstracts.tsv", index=False, sep='\t', header=False, quoting=csv.QUOTE_NONE)

entities = pd.read_csv("train_data/drugprot_training_entities.tsv", sep='\t', header=None, names=["pmid", "arg", "type", "start", "end", "name"], dtype={"pmid": str})
train_entities = entities[entities["pmid"].isin(ids_train)]
val_entities = entities[entities["pmid"].isin(ids_val)]
test_entities = entities[entities["pmid"].isin(ids_test)]
train_entities.to_csv("split_original/drugprot_training_entities.tsv", index=False, sep='\t', header=False, quoting=csv.QUOTE_NONE)
val_entities.to_csv("split_original/drugprot_development_entities.tsv", index=False, sep='\t', header=False, quoting=csv.QUOTE_NONE)
test_entities.to_csv("split_original/drugprot_test_entities.tsv", index=False, sep='\t', header=False, quoting=csv.QUOTE_NONE)

relations = pd.read_csv("train_data/drugprot_training_relations.tsv", sep='\t', header=None, names=["pmid", "type", "arg1", "arg2"], dtype={"pmid": str})
train_relations = relations[relations["pmid"].isin(ids_train)]
val_relations = relations[relations["pmid"].isin(ids_val)]
test_relations = relations[relations["pmid"].isin(ids_test)]
train_relations.to_csv("split_original/drugprot_training_relations.tsv", index=False, sep='\t', header=False, quoting=csv.QUOTE_NONE)
val_relations.to_csv("split_original/drugprot_development_relations.tsv", index=False, sep='\t', header=False, quoting=csv.QUOTE_NONE)
test_relations.to_csv("split_original/drugprot_test_relations.tsv", index=False, sep='\t', header=False, quoting=csv.QUOTE_NONE)
