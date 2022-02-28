from nltk.lm.smoothing import KneserNey
from sentence_transformers import SentenceTransformer, SentencesDataset, losses, models, util
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import LabelAccuracyEvaluator

from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from utils import transform_sentence, label_dict, get_training_split_on_pmid

train_batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_json("processed_dataset/train_dataset.json", orient="table")
df2 = pd.read_json("processed_dataset/development_dataset.json", orient="table")
df_all = pd.concat([df, df2], ignore_index=True)

ids_train, ids_val = get_training_split_on_pmid(df_all['pmid'].unique())

sentences, chemicals, genes = df_all["sentence"], df_all["chemical"], df_all["gene"]
df_all.loc[:, "sentence"] = transform_sentence(sentences, chemicals, genes)

df_all = df_all.drop(columns=["all_chemicals", "all_geneYs", "all_geneNs"])

df_add1 = pd.read_csv("train_add_Pubfull.csv", index_col=0)
df_add2 = pd.read_csv("dev_add_Pubfull.csv", index_col=0)
df_add = pd.concat([df_add1, df_add2], ignore_index=True)
df_add = df_add.rename(columns={"label": "relation type"})
df_add = df_add.drop(columns=["all_chemicals", "all_geneYs", "all_geneNs"])
df_add = df_add.astype({"pmid": "str"})
whole_df = pd.concat([df_all, df_add], ignore_index=True)

df_train = whole_df[whole_df['pmid'].isin(ids_train)]
df_val = whole_df[whole_df['pmid'].isin(ids_val)]

print("Sample size: train: {}, val: {}".format(len(df_train), len(df_val)))

relation_to_label_dict = {val: key for key, val in label_dict.items()}

relation_train = [relation_to_label_dict[relation] for relation in df_train["relation type"]]
relation_val = [relation_to_label_dict[relation] for relation in df_val["relation type"]]

df_train["label"] = relation_train
df_val["label"] = relation_val

label_list = set(df_train['label'])

train_examples = []
train_sentences = []
for idx, example in df_train.iterrows():
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

from torch import nn
import os
import sys

# out_features, margin, epochs, lr = sys.argv[1:]
out_features, margin, epochs, lr = 256, 5.0, 10, 2e-5

# out_features = os.environ.get("OUT_FEATURES", 256)
# margin = os.environ.get("MARGIN", 5)
# epochs = os.environ.get("EPOCHS", 10)
# lr = os.environ.get("LR", 2e-5)

out_features, epochs = map(int, [out_features, epochs])
margin = float(margin)
lr = float(lr)
print(out_features, margin, epochs, lr)
# DO THIS LATER LOSSES = [losses.BatchHardTripletLoss, losses.BatchSemiHardTripletLoss]
param_str = '_'.join(map(str, [out_features, margin, epochs, lr]))
print(param_str)


from sentence_transformers import SentenceTransformer, models

word_embedding_model = models.Transformer('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=out_features, activation_function=nn.Tanh())


tokens = ["chem_name", "gene_name"]
word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.BatchHardTripletLoss(model=model, margin=margin)

val_dataset = SentencesDataset(val_examples, model)
val_dataloader = DataLoader(val_dataset, batch_size=1)
evaluator = LabelAccuracyEvaluator(dataloader=val_dataloader)

from sentence_transformers.evaluation import SentenceEvaluator
class MyCustomEvaluator(SentenceEvaluator):
    def __init__(self, train_data, test_data, k_in_NN=1000):
        self.train_data = train_data
        self.train_true_labels = [example.label for example in train_data]
        self.train_sentences = [example.texts[0] for example in train_data]

        self.test_data = test_data
        self.test_true_labels = [example.label for example in test_data]
        self.test_sentences = [example.texts[0] for example in test_data]

        self.k_in_NN = k_in_NN # number of Nearest Neighbors

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        """
        This is called during training to evaluate the model.
        It returns a score for the evaluation with a higher score indicating a better result.

        :param model:
            the model to evaluate
        :param output_path:
            path where predictions and metrics are written to
        :param epoch
            the epoch where the evaluation takes place.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation on test data.
        :param steps
            the steps in the current epoch at time of the evaluation.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation at the end of the epoch.
        :return: a score for the evaluation with a higher score indicating a better result
        """
        train_vecs = model.encode(self.train_sentences, convert_to_numpy=True)
        test_vecs = model.encode(self.test_sentences, convert_to_numpy=True)

        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_neighbors=self.k_in_NN)
        neigh.fit(train_vecs, self.train_true_labels)
        pred_test_labels = neigh.predict(test_vecs)
        true_classes = [label_dict[true_class] for true_class in self.test_true_labels]
        pred_classes = [label_dict[pred_class] for pred_class in pred_test_labels]
        from sklearn.metrics import f1_score
        labels = set(label_dict.values())
        labels.remove("NOT")
        f1 = f1_score(true_classes, pred_classes, labels=list(labels), average="micro")
        print(f1)
        return f1

my_evaluator = MyCustomEvaluator(train_data=train_examples, test_data=val_examples, k_in_NN=1000)
save_path = 'auto_sbert_microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_'+param_str

model.fit(train_objectives=[(train_dataloader, train_loss)], evaluator=my_evaluator, epochs=epochs, warmup_steps=100, optimizer_class=AdamW, optimizer_params={'lr': lr}, output_path=save_path, save_best_model=True)
train_vecs = model.encode(train_sentences, convert_to_numpy=True)
train_labels = [train_example.label for train_example in train_examples]
model.save('sbert_microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_'+param_str)

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_vecs, train_labels)

val_vecs = model.encode(val_sentences, convert_to_numpy=True)
print(neigh.predict(val_vecs))
print(sum(neigh.predict(val_vecs)))



# look at errors which class we make wrong predictions
# do clustering on our own data
# 
# use cluster membership as input
# run clustering, for different clusters check performance
# 0.72425, after reshuffing w/random = 100
# 0.735 after augmentation
