# -*-coding:utf-8-*-
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset_train
from vocab import Vocab

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

train_batch_size = 512
test_batch_size = 500
# voc_model = pickle.load(open("./models/vocab.pkl", "rb"))
sequence_max_len = 100
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

        self.fc = nn.Linear(sequence_max_len * 200, 2)

    def forward(self, input):
        """
        :param input:[batch_size,max_len]
        :return:
        """
        input_embeded = self.embedding(input)  # input embeded :[batch_size,max_len,200]

        # 变形
        input_embeded_viewed = input_embeded.view(input_embeded.size(0), -1)

        # 全连接
        out = self.fc(input_embeded_viewed)
        return F.log_softmax(out, dim=-1)


def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def train(imdb_model, imdb_dataset, epoch):
    train_dataloader = get_dataloader(imdb_dataset, train=True)
    optimizer = Adam(imdb_model.parameters())
    best_loss = float('inf')

    history = {
        "loss": []
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
            bar.set_description("epoch:{}  idx:{}   loss:{:.6f}".format(i, idx, loss.item()))

        avg_loss = epoch_loss / len(train_dataloader)
        history["loss"].append(avg_loss)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(imdb_model, "./models/fc_model_best.pkl")
            torch.save(imdb_model.state_dict(), "./models/fc_model_state_dict_best.pkl")
            print(f"Best model saved at epoch {i} with avg_loss {avg_loss:.6f}")

    return history



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def test(model, imdb_dataset):
    """
    验证模型，返回准确率、精确率、召回率、F1分数
    """
    model.eval()
    dataloader = get_dataloader(imdb_dataset, train=False)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for reviews, labels in dataloader:
            reviews = reviews.to(model.embedding.weight.device)
            outputs = model(reviews)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return acc, precision, recall, f1


import os
import matplotlib.pyplot as plt

def plot_metrics(history, save_dir="./result", filename="metrics_plot.png"):
    """
    绘制训练过程中的loss、accuracy、precision、recall、f1的折线图，并保存到result目录
    """
    # 如果保存目录不存在，则创建
    os.makedirs(save_dir, exist_ok=True)

    # 生成图片路径
    save_path = os.path.join(save_dir, filename)

    epochs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(10, 6))

    # 绘制 loss
    plt.plot(epochs, history["loss"], label="Loss", marker='o')

    # 绘制 accuracy, precision, recall, f1
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

    # 保存图片到 result 文件夹
    plt.savefig(save_path)
    print(f"📊 训练过程折线图已保存到: {save_path}")

    plt.close()  # 防止内存泄漏
