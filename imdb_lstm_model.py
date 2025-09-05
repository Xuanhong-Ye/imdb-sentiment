# -*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset_train
from vocab import Vocab

train_batch_size = 512
test_batch_size = 128
# voc_model = pickle.load(open("./models/vocab.pkl", "rb"))
sequence_max_len = 200
Vocab()


def collate_fn(batch):
    """
    对batch数据进行处理
    :param batch: [一个getitem的结果，getitem的结果,getitem的结果]
    :return: 元组
    """
    reviews, labels = zip(*batch)
    # reviews = torch.LongTensor([voc_model.transform(i, max_len=sequence_max_len) for i in reviews])
    reviews = torch.LongTensor(reviews)
    labels = torch.LongTensor(labels)
    return reviews, labels


def get_dataset(train=True):
    return dataset_train.ImdbDataset(train)


def get_dataloader(imdb_dataset, train=True):
    batch_size = train_batch_size if train else test_batch_size
    return DataLoader(imdb_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


class ImdbModel(nn.Module):
    def __init__(self, num_embeddings, padding_idx):
        super(ImdbModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=200, padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=200, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True,
                            dropout=0.5)
        self.fc1 = nn.Linear(64 * 2, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, input):
        """
        :param input:[batch_size,max_len]
        :return:
        """
        input_embeded = self.embedding(input)  # input embeded :[batch_size,max_len,200]

        output, (h_n, c_n) = self.lstm(input_embeded)  # h_n :[4,batch_size,hidden_size]
        # out :[batch_size,hidden_size*2]
        out = torch.cat([h_n[-1, :, :], h_n[-2, :, :]], dim=-1)  # 拼接正向最后一个输出和反向最后一个输出

        # 进行全连接
        out_fc1 = self.fc1(out)
        # 进行relu
        out_fc1_relu = F.relu(out_fc1)

        # 全连接
        out_fc2 = self.fc2(out_fc1_relu)  # out :[batch_size,2]
        return F.log_softmax(out_fc2, dim=-1)


def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def train(imdb_model, imdb_dataset, test_dataset=None, epoch=1):
    train_dataloader = get_dataloader(imdb_dataset, train=True)
    optimizer = Adam(imdb_model.parameters())
    best_loss = float('inf')  # 初始化最小loss为正无穷

    # 用于记录折线图数据
    history = {
        "loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    for i in range(epoch):
        imdb_model.train()
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        epoch_loss = 0
        for idx, (data, target) in enumerate(bar):
            optimizer.zero_grad()
            data = data.to(device())
            target = target.to(device())
            output = imdb_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            bar.set_description(f"epoch:{i}  idx:{idx}   loss:{loss.item():.6f}")

        avg_loss = epoch_loss / len(train_dataloader)
        history["loss"].append(avg_loss)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(imdb_model, "./models/lstm_model_best.pkl")
            torch.save(imdb_model.state_dict(), "./models/lstm_model_state_dict_best.pkl")
            print(f"Best model saved at epoch {i} with avg_loss {avg_loss:.6f}")

        # 如果提供了测试集，则每轮训练后测试
        if test_dataset is not None:
            acc, precision, recall, f1 = test(imdb_model, test_dataset, return_metrics=True)
            history["accuracy"].append(acc)
            history["precision"].append(precision)
            history["recall"].append(recall)
            history["f1"].append(f1)

    return history



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def test(imdb_model, imdb_dataset, return_metrics=False):
    imdb_model.eval()
    test_dataloader = get_dataloader(imdb_dataset, train=False)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in tqdm(test_dataloader):
            data = data.to(device())
            target = target.to(device())
            output = imdb_model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    if return_metrics:
        return acc, precision, recall, f1


import os
import matplotlib.pyplot as plt

def plot_metrics(history, save_dir="./result", filename="lstm_metrics_plot.png"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    epochs = range(1, len(history["loss"]) + 1)
    plt.figure(figsize=(10, 6))

    # 绘制 loss
    plt.plot(epochs, history["loss"], label="Loss", marker='o')
    # 绘制其他指标
    plt.plot(epochs, history["accuracy"], label="Accuracy", marker='o')
    plt.plot(epochs, history["precision"], label="Precision", marker='o')
    plt.plot(epochs, history["recall"], label="Recall", marker='o')
    plt.plot(epochs, history["f1"], label="F1-score", marker='o')

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Loss and Metrics per Epoch")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f" 折线图已保存: {save_path}")
