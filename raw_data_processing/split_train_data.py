import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Run make_dataset.py first
df = pd.read_csv("train_dataset.csv", index_col=0)

ids_train, ids_test = train_test_split(df['pmid'].unique(), test_size=0.20, random_state=1)
ids_val, ids_test = train_test_split(ids_test, test_size=0.5, random_state=1)

df_train = df[df['pmid'].isin(ids_train)]
df_val = df[df['pmid'].isin(ids_val)]
df_test = df[df['pmid'].isin(ids_test)]

if not os.path.isdir("split_data"):
    os.mkdir("split_data") 

df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

df_train.to_csv("split_data/train_dataset.csv")
df_val.to_csv("split_data/val_dataset.csv")
df_test.to_csv("split_data/test_dataset.csv")
