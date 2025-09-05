import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np

class ImdbDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=200):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, pretrained_model='./models/bert-base-uncased', num_labels=2):
        super().__init__()
        # 加载 safetensors 权重
        self.bert = BertModel.from_pretrained(pretrained_model, use_safetensors=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)



    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.fc(pooled)

def get_tokenizer():
    return BertTokenizer.from_pretrained('./models/bert-base-uncased')

def get_dataloader(texts, labels, tokenizer, batch_size=16, shuffle=True, max_len=256):
    dataset = ImdbDataset(texts, labels, tokenizer, max_len=max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            pred = outputs.argmax(dim=1).cpu().numpy()
            preds.extend(pred)
            trues.extend(labels.cpu().numpy())
    return preds, trues