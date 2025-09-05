import os
import pickle

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_vocab import ImdbDataset, collate_fn
from vocab import Vocab


def get_dataloader(train=True):
    imdb_dataset = ImdbDataset(train, sequence_max_len=100)
    my_dataloader = DataLoader(imdb_dataset, batch_size=200, shuffle=True, collate_fn=collate_fn)
    return my_dataloader


if __name__ == '__main__':

    ws = Vocab()
    dl_train = get_dataloader(True)
    for reviews in tqdm(dl_train, total=len(dl_train)):
        for sentence in reviews:
            ws.fit(sentence)

    ws.build_vocab()
    print(len(ws))
    if not os.path.exists("./models"):
        os.makedirs("./models")
    # 保存词典
    with open("./models/vocab.pkl", "wb") as f:
        pickle.dump(ws, f)

    # ====== 测试词典 ======
    # 重新加载词典
    with open("./models/vocab.pkl", "rb") as f:
        vocab_loaded = pickle.load(f)
    # 打印词典大小
    print("词典大小:", len(vocab_loaded))
    # 打印前100个词
    print("前10个词:", list(vocab_loaded.dict.keys())[:100])
    # 测试分词转换
    test_sentence = "This is a good movie!"
    tokens = test_sentence.lower().split()
    print("转换结果:", vocab_loaded.transform(tokens, max_len=100))