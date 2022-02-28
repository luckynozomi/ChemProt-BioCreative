#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np
from collections import defaultdict

from utils import calculate_f1

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
    

def data_loader(df, tokenizer, max_len, batch_size):
    ds = SentenceDataset(
        sentences=df.sentence.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
        )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
        )


class RelationClassifier(nn.Module):
    
    def __init__(self, n_classes, bert_pretrain_path):
        super(RelationClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_pretrain_path, return_dict=False)
        self.drop = nn.Dropout(p=0.1)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
        output = self.drop(pooled_output)
        return self.out(output), pooled_output
    
    def train_epoch(self, data_loader, loss_fn, optimizer, device, scheduler):
        model = self.train()
        losses = []
        correct_predictions = 0
        predictions = []
        prediction_probs = []
        real_values = []
        
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)
            
            outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(labels)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        return calculate_f1(real_values, predictions), np.mean(losses)

    def eval_model(self, data_loader, loss_fn, device):
        model = self.eval()
        losses = []
        correct_predictions = 0
        predictions = []
        prediction_probs = []
        real_values = []
        
        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                labels = d["label"].to(device)
                
                outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                probs = F.softmax(outputs, dim=1)
                loss = loss_fn(outputs, labels)
                
                correct_predictions += torch.sum(preds == labels)
                losses.append(loss.item())
                predictions.extend(preds)
                prediction_probs.extend(probs)
                real_values.extend(labels)
        
        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        return calculate_f1(real_values, predictions), np.mean(losses)


    def get_predictions(self, data_loader, device):
        model = self.eval()
        sentence_texts = []
        predictions = []
        prediction_probs = []
        real_values = []
        vecs = []
        with torch.no_grad():
            for d in data_loader:
                texts = d["sentence"]
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                labels = d["label"].to(device)
                
                outputs, vec = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                probs = F.softmax(outputs, dim=1)
                
                sentence_texts.extend(texts)
                predictions.extend(preds)
                prediction_probs.extend(probs)
                real_values.extend(labels)
                vecs.extend(vec)
        
        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        vecs = torch.stack(vecs).cpu()
        return sentence_texts, predictions, prediction_probs, real_values, vecs


def main_train_fn(model, train_data, val_data, n_epochs, loss_fn, optimizer, device, scheduler, save_path):
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1}/{n_epochs}')
        print('-' * 10)
        
        train_acc, train_loss = model.train_epoch(
            train_data,
            loss_fn,
            optimizer,
            device,
            scheduler
            )
        print(f'Train loss {train_loss} f1 {train_acc}')
        
        val_acc, val_loss = model.eval_model(
            val_data,
            loss_fn,
            device,
            )
        print(f'Val   loss {val_loss} f1 {val_acc}')
        print()
        
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), save_path)
            best_accuracy = val_acc

    return history