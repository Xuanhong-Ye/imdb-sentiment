import torch

import imdb_lstm_model


if __name__ == '__main__':
    train_dataset = imdb_lstm_model.get_dataset(train=True)
    test_dataset = imdb_lstm_model.get_dataset(train=False)

    imdb_model = imdb_lstm_model.ImdbModel(
        train_dataset.get_num_embeddings(),
        train_dataset.get_padding_idx()
    ).to(imdb_lstm_model.device())

    # 训练并记录折线图数据
    history = imdb_lstm_model.train(imdb_model, train_dataset, test_dataset=test_dataset, epoch=10)

    # 绘制折线图
    imdb_lstm_model.plot_metrics(history)


# if __name__ == '__main__':
#     # 分别获取训练集和测试集
#     train_dataset = imdb_lstm_model.get_dataset(train=True)
#     test_dataset = imdb_lstm_model.get_dataset(train=False)
#     # 训练
#     imdb_model = imdb_lstm_model.ImdbModel(
#         train_dataset.get_num_embeddings(),
#         train_dataset.get_padding_idx()
#     ).to(imdb_lstm_model.device())
#     imdb_lstm_model.train(imdb_model, train_dataset, 1)
#     # 测试（只加载参数，不会有warning）
#     path_lstm_model = "./models/lstm_model_state_dict.pkl"
#     test_model = imdb_lstm_model.ImdbModel(
#         train_dataset.get_num_embeddings(),
#         train_dataset.get_padding_idx()
#     ).to(imdb_lstm_model.device())
#     test_model.load_state_dict(torch.load(path_lstm_model, weights_only=True))
#     imdb_lstm_model.test(test_model, test_dataset)
    
#     train_dataset = imdb_lstm_model.get_dataset(train=True)
#     test_dataset = imdb_lstm_model.get_dataset(train=False)
#     print("Train sample:", train_dataset.total_path[0])
#     print("Test sample:", test_dataset.total_path[0])