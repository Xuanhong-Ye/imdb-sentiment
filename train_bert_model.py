import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imdb_bert_model import BertClassifier, get_tokenizer, get_dataloader, train, evaluate
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_metrics(history, save_path=None):
    epochs = range(1, len(history['loss']) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['loss'], label='Loss', marker='o')
    plt.plot(epochs, history['accuracy'], label='Accuracy', marker='o')
    plt.plot(epochs, history['precision'], label='Precision', marker='o')
    plt.plot(epochs, history['recall'], label='Recall', marker='o')
    plt.plot(epochs, history['f1'], label='F1-score', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("BERT Training Metrics")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    # 加载数据
    train_data = np.load('./data/download/data.npz', allow_pickle=True)
    test_data = np.load('./data/download/test.npz', allow_pickle=True)
    texts_train = train_data['a']
    labels_train = train_data['b'].astype(float).astype(int)
    texts_test = test_data['a']
    labels_test = test_data['b'].astype(float).astype(int)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = get_tokenizer()
    max_len = 200  # 可以修改 max_len

    train_loader = get_dataloader(texts_train, labels_train, tokenizer, batch_size=16, shuffle=True, max_len=max_len)
    test_loader = get_dataloader(texts_test, labels_test, tokenizer, batch_size=16, shuffle=False, max_len=max_len)

    model = BertClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # 保存每轮的指标
    history = {
        'loss': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    epochs = 10  # 可以修改为你需要的轮次
    for epoch in range(epochs):
        # 训练
        avg_loss = train(model, train_loader, optimizer, device)
        history['loss'].append(avg_loss)

        # 测试
        preds, trues = evaluate(model, test_loader, device)
        acc = accuracy_score(trues, preds)
        precision = precision_score(trues, preds)
        recall = recall_score(trues, preds)
        f1 = f1_score(trues, preds)

        history['accuracy'].append(acc)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)

        print(f"Epoch {epoch+1}/{epochs} | Loss={avg_loss:.4f} | "
              f"Acc={acc:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")

    # 创建 result 文件夹保存图像
    os.makedirs('./result', exist_ok=True)
    plot_metrics(history, save_path='./result/bert_training_metrics.png')
