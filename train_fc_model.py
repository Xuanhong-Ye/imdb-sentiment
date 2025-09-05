import torch
from imdb_fc_model import get_dataset, ImdbModel, train, test, device, plot_metrics

if __name__ == '__main__':
    # 获取训练集和测试集
    train_dataset = get_dataset(train=True)
    test_dataset = get_dataset(train=False)

    # 创建模型
    imdb_model = ImdbModel(train_dataset.get_num_embeddings(), train_dataset.get_padding_idx()).to(device())

    epochs = 10
    history = {
        "loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    # 循环训练，每一轮后进行测试
    for epoch in range(epochs):
        # 训练 1 个 epoch
        train_history = train(imdb_model, train_dataset, 1)
        history["loss"].extend(train_history["loss"])

        # 测试当前模型
        acc, precision, recall, f1 = test(imdb_model, test_dataset)
        history["accuracy"].append(acc)
        history["precision"].append(precision)
        history["recall"].append(recall)
        history["f1"].append(f1)

        print(f"Epoch {epoch+1}/{epochs}: Loss={train_history['loss'][-1]:.4f}, "
              f"Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    # 最后绘制折线图
    plot_metrics(history)
