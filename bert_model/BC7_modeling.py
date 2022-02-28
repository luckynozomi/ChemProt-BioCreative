#!/usr/bin/env python
# coding: utf-8

from concurrent.futures import process
from random import randrange

import pandas as pd
import json
import torch
from torch import nn
import transformers
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import defaultdict
import numpy as np

from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformers.logging.set_verbosity_error()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

label_list = ['NOT', 'INDIRECT-UPREGULATOR', 'DIRECT-REGULATOR', 'AGONIST-INHIBITOR', 'PART-OF', 'PRODUCT-OF', 'AGONIST-ACTIVATOR', 'SUBSTRATE_PRODUCT-OF', 'AGONIST', 'ACTIVATOR', 'SUBSTRATE', 'ANTAGONIST', 'INHIBITOR', 'INDIRECT-DOWNREGULATOR']
label_dict = {idx: val for idx, val in enumerate(label_list)}

def transform_sentence(entry):
    gene = entry["gene"]
    chemical = entry["chemical"]
    sent = entry["sentence"]
    all_poses = [chemical[2:4]+["CHEMICAL"]] + [gene[2:4]+["GENE"]]
    all_poses.sort(key=lambda i: i[0], reverse=True)
    for start, end, e_type in all_poses:
        sent = sent[0:start] + e_type + sent[end:]
    sent = chemical[4] + ', ' + gene[4] + ', ' + sent
    return sent

def process_data(dataframe):
    dataframe["label"] = dataframe["relation type"]
    for key, label in label_dict.items():
        dataframe['label'] = dataframe['label'].replace(label, key)

    dataframe["text"] = [transform_sentence(entry) for _, entry in dataframe.iterrows()]
    return dataframe

df_train = pd.read_json("BC7_dataset/train_dataset.json", orient="table")
df_val = pd.read_json("BC7_dataset/development_dataset.json", orient="table")
df_test = pd.read_json("BC7_dataset/test_dataset.json", orient="table")

df_train = process_data(df_train)
df_val = process_data(df_val)
df_test = process_data(df_test)

# Model Parameters
PRE_TRAINED_MODEL_NAME = "/data/xin.sui/litcoin_phase2/pre_trained_models/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf"
MAX_LEN = 384
import sys
# Training Parameters
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = float(sys.argv[1])
print("learning rate: ", LEARNING_RATE)
SAVE_PATH = "roberta_large_{}".format(LEARNING_RATE)
SAVE_PATH = "fine_tuned_models/{}".format(SAVE_PATH)

# https://huggingface.co/blog/zero-deepspeed-fairscale
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
tokens = ["CHEMICAL", "GENE"]
tokenizer.add_tokens(tokens, special_tokens=True)

class SentenceDataset(Dataset):
    
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'sentence': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

train_data = SentenceDataset(
    sentences=df_train.text.to_numpy(),
    labels=df_train.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
val_data = SentenceDataset(
    sentences=df_val.text.to_numpy(),
    labels=df_val.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
test_data = SentenceDataset(
    sentences=df_test.text.to_numpy(),
    labels=df_test.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

from transformers import AutoModelForSequenceClassification
n_classes = len(label_dict)
model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=n_classes)

if model.base_model_prefix == "roberta":
    model.roberta.resize_token_embeddings(len(tokenizer))

elif model.base_model_prefix == "bert":
    model.bert.resize_token_embeddings(len(tokenizer))
else:
    raise NotImplementedError

model = model.to(device)

from transformers import Trainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=SAVE_PATH,
    num_train_epochs=EPOCHS,
    evaluation_strategy='epoch',
    learning_rate=LEARNING_RATE,
    save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
    save_strategy='epoch',
    metric_for_best_model = 'f1',
    warmup_ratio=0.1,
    per_device_train_batch_size=BATCH_SIZE, # 6231 -> 3633MB
    per_device_eval_batch_size=BATCH_SIZE,
    # gradient_accumulation_steps=4,
    # eval_accumulation_steps=4,
    load_best_model_at_end=True,
    # deepspeed="dsconfig.json",
    # gradient_checkpointing=True,
    fp16=False  # 6903 -> 6231
)

# https://stackoverflow.com/questions/69087044/early-stopping-in-bert-trainer-instances
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
def compute_f1(evalprediction_instance):
    predictions, label_ids = evalprediction_instance.predictions, evalprediction_instance.label_ids
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(label_ids, predictions)
    f1 = f1_score(label_ids, predictions, labels=range(1, len(label_dict)), average='micro')
    return {"accuracy": accuracy, "f1": f1}

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_data, 
    eval_dataset=val_data,
    compute_metrics=compute_f1
)

trainer.train()
print(trainer.evaluate(eval_dataset=test_data))
trainer.save_model("{}/{}".format(SAVE_PATH, "final_model"))
tokenizer.save_pretrained("{}/{}".format(SAVE_PATH, "tokenizer"))
